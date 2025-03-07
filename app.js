// app.js - Main Server File
import express from 'express';
import http from 'http';
import { Server } from 'socket.io';
import cors from 'cors';
import { rateLimit } from 'express-rate-limit';
import { MongoClient } from 'mongodb';
import dotenv from 'dotenv';
import { fileURLToPath } from 'url';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { createWriteStream } from 'fs';
import { stringify } from 'csv-stringify';
import sanitizeHtml from 'sanitize-html'; // equivalent to bleach
import fetch from 'node-fetch';
import { getResponse, getCurrentModel } from './llm_query.js';
import { ChromaDB } from './db_singleton.js';

// Load environment variables
dotenv.config();

// Paths and environment setup
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Environment variables
const SLACK_HOOK = process.env.SLACK_HOOK;
const FILE_PASSWORD = process.env.FILE_PASSWORD;
const MONGO_CONVERSATION_DB = process.env.MONGO_CONVERSATION_DB || 'ai_conversations';
const APOS_MONGODB_URI = process.env.APOS_MONGODB_URI || 'mongodb://localhost:27017';

// Initialize Express app and socket.io
const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: '*',
    methods: ['GET', 'POST']
  }
});

// Apply middleware
app.use(cors());
app.use(express.json());

// Initialize MongoDB connection
let mongoClient;
let conversationCollection;

const connectToMongo = async () => {
  try {
    mongoClient = new MongoClient(APOS_MONGODB_URI);
    await mongoClient.connect();
    console.log('Connected to MongoDB.');
    
    const db = mongoClient.db(MONGO_CONVERSATION_DB);
    conversationCollection = db.collection('conversations');
  } catch (error) {
    console.error(`Failed to connect to MongoDB. Error: ${error}`);
  }
};

// Initialize the rate limiter
const limiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 2, // 2 requests per minute
  standardHeaders: true,
  legacyHeaders: false,
});

// Initialize ChromaDB instance
let chromaInstance;
let retriever;

const initChroma = async () => {
  try {
    chromaInstance = await ChromaDB.getInstance();
    retriever = chromaInstance.retriever;
    console.log('ChromaDB initialized successfully');
  } catch (error) {
    console.error(`Failed to initialize ChromaDB: ${error}`);
  }
};

// Session storage (in-memory for simplicity, could be moved to Redis for production)
const sessions = {};

// Helper functions
const logToSlack = async (sessionId, question, answer, modelName) => {
  const logToSlackEnabled = process.env.LOG_TO_SLACK === 'true';
  console.log(`Slack logging enabled: ${logToSlackEnabled} at ${SLACK_HOOK}`);
  
  if (logToSlackEnabled && SLACK_HOOK) {
    const slackData = {
      text: `User session ID: ${sessionId}\nTime: ${new Date().toISOString()}\nModel: ${modelName}\nQuestion: ${question}\nAnswer: ${answer}`
    };

    try {
      const response = await fetch(SLACK_HOOK, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(slackData)
      });

      if (response.status !== 200) {
        throw new Error(`Request to slack returned an error ${response.status}, the response is: ${await response.text()}`);
      }
    } catch (error) {
      console.error('Error sending to Slack:', error);
    }
  } else {
    console.log(`Slack logging disabled with model ${modelName}.`);
  }
};

const logToMongo = async (sessionId, question, answer, modelName) => {
  if (!conversationCollection) return;
  
  const conversationData = {
    session_id: sessionId,
    query: question,
    answer: answer,
    model: modelName,
    timestamp: new Date()
  };
  
  try {
    await conversationCollection.insertOne(conversationData);
  } catch (error) {
    console.error('Error logging to MongoDB:', error);
  }
};

// Socket.io event handlers
io.on('connection', (socket) => {
  let userSessionId = socket.handshake.query.user_session_id;
  
  if (!userSessionId) {
    userSessionId = uuidv4();
  }
  
  sessions[socket.id] = {
    userSessionId,
    requestInProgress: false
  };
  
  socket.emit('session_id', { user_session_id: userSessionId });
  
  socket.on('query', async (data) => {
    if (sessions[socket.id].requestInProgress) {
      socket.emit('error', { message: 'Please wait for the current response.' });
      return;
    }
    
    sessions[socket.id].requestInProgress = true;
    
    const sessionId = sessions[socket.id].userSessionId;
    const query = sanitizeHtml(data.query);
    const index = data.index;
    
    console.log(`Message: ${query}, Session ID: ${sessionId}`);
    
    try {
      // Get the response
      const response = await getResponse(query, retriever, sessionId);
      const modelName = getCurrentModel();
      
      // Send the response
      socket.emit('answer', { text: response, index });
      
      // Log the response
      await logToSlack(sessionId, query, response, modelName);
      await logToMongo(sessionId, query, response, modelName);
    } catch (error) {
      console.error('Error processing query:', error);
      socket.emit('error', { message: 'An error occurred while processing your query.' });
    } finally {
      sessions[socket.id].requestInProgress = false;
    }
  });
  
  socket.on('clear_session', () => {
    delete sessions[socket.id];
    console.log('Session cleared.');
    socket.disconnect(true);
  });
  
  socket.on('disconnect', () => {
    delete sessions[socket.id];
  });
});

// Express routes
app.get('/', (req, res) => {
  res.status(200).send('OK');
});

app.get('/export_conversations', async (req, res) => {
  // Check the password from the query parameter
  const password = req.query.password;
  if (password !== FILE_PASSWORD) {
    return res.status(403).send('Forbidden');
  }

  if (!conversationCollection) {
    return res.status(500).send('MongoDB not connected');
  }

  try {
    // Query the MongoDB collection for all conversations
    const conversations = await conversationCollection.find({}).toArray();
    
    // Create a CSV string
    const csvStream = stringify({ header: true, columns: [
      'session_id', 'query', 'answer', 'model', 'timestamp'
    ]});
    
    // Write header
    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', 'attachment; filename=conversations.csv');
    
    // Pipe the CSV data to the response
    csvStream.pipe(res);
    
    // Write the data
    for (const conversation of conversations) {
      csvStream.write([
        conversation.session_id || '',
        conversation.query || '',
        conversation.answer || '',
        conversation.model || 'gpt-4o',
        conversation.timestamp ? conversation.timestamp.toISOString() : ''
      ]);
    }
    
    csvStream.end();
  } catch (error) {
    console.error('Error exporting conversations:', error);
    res.status(500).send('Error exporting conversations');
  }
});

// Initialize and start the server
const PORT = process.env.PORT || 3000;

const startServer = async () => {
  await connectToMongo();
  await initChroma();
  
  server.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on port ${PORT}`);
  });
};

startServer().catch(console.error);

export default app;