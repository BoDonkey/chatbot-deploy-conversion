# ApostropheCMS AI Chatbot (JavaScript Version)

This is a JavaScript implementation of the ApostropheCMS AI chatbot, converted from the original Python version. It maintains compatibility with ChromaDB, LangChain, and model selection capabilities.

## Features

- Real-time chat using Socket.IO
- Integration with ChromaDB for vector storage
- Support for OpenAI and Anthropic models
- MongoDB for conversation storage
- Rate limiting to prevent abuse
- Slack integration for logging

## Prerequisites

- Node.js 18 or higher
- MongoDB
- ChromaDB
- OpenAI API key and/or Anthropic API key

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/apos-chatbot-js.git
   cd apos-chatbot-js
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Create a `.env` file based on the provided sample:
   ```bash
   cp .env.example .env
   ```

4. Update the `.env` file with your API keys and configuration.

## ChromaDB Setup

The system requires a ChromaDB instance with your ApostropheCMS documentation loaded. If you're migrating from the Python version, you can use the same ChromaDB directory.

### Setting Up a New ChromaDB

If you need to create a new ChromaDB instance, use the [rag_database_creation repo](https://github.com/apostrophecms/rag_database_creation)

## Running the Application

Start the server:

```bash
npm start
```

For development with auto-reload:

```bash
npm run dev
```

The server will run on port 3000 by default (or the port specified in your .env file).

## API Endpoints

- `GET /`: Health check endpoint
- `GET /export_conversations?password=YOUR_PASSWORD`: Export all conversations as CSV

## Socket.IO Events

### Client to Server
- `query`: Send a question to the chatbot
- `clear_session`: Clear the current session
- `disconnect`: Disconnect from the server

### Server to Client
- `session_id`: Sends the assigned session ID
- `answer`: Sends the chatbot's response
- `error`: Sends error messages

## Exporting Conversations

To export all conversations as a CSV file:

```
http://localhost:3000/export_conversations?password=your-export-password
```

Replace `your-export-password` with the value set in your .env file.

## Customization

### Changing the LLM Model

Edit the `CHAT_MODEL` in your .env file:
- `ChatOpenAI`: Uses OpenAI models (default is gpt-4o)
- `ChatAnthropic`: Uses Anthropic models (default is claude-3-5-sonnet-20240620)

### Modifying the Prompt

To change how the AI responds, edit the `DEFAULT_TEMPLATE` in the `llm_query.js` file.

## Troubleshooting

### ChromaDB Connection Issues
- Ensure your ChromaDB is properly set up with the correct collection name ("langchain")
- Check the `CHROMA_PATH` in your .env file

### MongoDB Connection Issues
- Verify your MongoDB server is running
- Check the `APOS_MONGODB_URI` in your .env file

### API Key Issues
- Make sure you've set the correct API keys in your .env file
- Check for API usage limits or restrictions