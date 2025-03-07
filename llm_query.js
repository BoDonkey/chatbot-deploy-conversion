// llm_query.js
import { ChatOpenAI } from '@langchain/openai';
import { ChatAnthropic } from '@langchain/anthropic';
import { OpenAIEmbeddings } from '@langchain/openai';
import { ChatMessageHistory } from '@langchain/community/chat_message_histories/in_memory';
import { 
  createHistoryAwareRetriever, 
  createRetrieval, 
  createStuffDocuments
} from 'langchain/chains/combine_documents';
import { RunnableWithMessageHistory } from '@langchain/core/runnables';
import { ChatPromptTemplate, MessagesPlaceholder, PromptTemplate } from '@langchain/core/prompts';
import dotenv from 'dotenv';
import { ChromaDB } from './db_singleton.js';

dotenv.config();

// Global store for chat histories
const chatHistoriesStore = {};
let currentModel = null;

// Function to calculate cosine similarity manually
function cosineSimilarity(vectorA, vectorB) {
  if (vectorA.length !== vectorB.length) {
    throw new Error('Vectors must have the same dimensions');
  }
  
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  
  for (let i = 0; i < vectorA.length; i++) {
    dotProduct += vectorA[i] * vectorB[i];
    normA += vectorA[i] * vectorA[i];
    normB += vectorB[i] * vectorB[i];
  }
  
  if (normA === 0 || normB === 0) {
    return 0;
  }
  
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Function to check similarity between questions
async function areQuestionsSimilar(question1, question2, threshold = 0.85) {
  const embeddings = new OpenAIEmbeddings();
  const embedding1 = await embeddings.embedQuery(question1);
  const embedding2 = await embeddings.embedQuery(question2);
  const similarity = cosineSimilarity(embedding1, embedding2);
  return similarity >= threshold;
}

// Retrieve or initialize chat history for a session
function getSessionHistory(sessionId) {
  if (!chatHistoriesStore[sessionId]) {
    chatHistoriesStore[sessionId] = new ChatMessageHistory();
  }
  return chatHistoriesStore[sessionId];
}

// Initialize and load your LLM, RAG DB, and conversational memory here
async function setupLlmAndDb(retriever) {
  // Get model choice from environment
  const modelChoice = process.env.CHAT_MODEL || 'ChatOpenAI';

  // Initialize the actual LLM
  let chat;
  if (modelChoice === 'ChatOpenAI') {
    chat = new ChatOpenAI({
      temperature: 0,
      modelName: 'gpt-4o'
    });
    currentModel = 'gpt-4o';
  } else {
    chat = new ChatAnthropic({
      temperature: 0.0,
      modelName: 'claude-3-5-sonnet-20240620'
    });
    currentModel = 'claude-3-5-sonnet-20240620';
  }

  // Contextualize question
  const contextualizeQSystemPrompt = `
    Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Match the language of the question or chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
  `;

  const contextualizeQPrompt = ChatPromptTemplate.fromMessages([
    ['system', contextualizeQSystemPrompt],
    new MessagesPlaceholder('chat_history'),
    ['human', '{input}'],
  ]);

  // Create history-aware retriever
  const historyAwareRetriever = await createHistoryAwareRetriever({
    llm: chat,
    retriever,
    rephrasePrompt: contextualizeQPrompt,
  });

  // Define the custom prompt template
  const DEFAULT_TEMPLATE = `
You are a senior developer with extensive expertise in Node.js, Express.js, Nunjucks, Vue.js, and the ApostropheCMS ecosystem (version 3 and above). Your main responsibility is to assist junior developers by providing insightful answers to their questions about developing within the ApostropheCMS framework. Utilize the RAG database documents in the context below to inform your answers. Ensure that your responses are comprehensive and directly applicable to the development practices within the newest ApostropheCMS context. When crafting answers, please adhere to the guidelines below and return the response in markdown format:
1. Relevance to ApostropheCMS Development: Only respond to inquiries that pertain to developing for ApostropheCMS. If a question falls outside this domain, kindly inform the user that it is beyond the scope of your expertise.
2. Attempt to be as concise as possible. Users should primarily be directed to the ApostropheCMS documentation for detailed information.
3. Documentation Links: Provide the top 2-3 unique links to relevant ApostropheCMS documentation or extension pages from the supplied URL. If no documentation exists, inform the user.
4. ALWAYS use ESM syntax by default, unless the user specifically asks for CommonJS (CJS) syntax.
5. ApostropheCMS Version: Ensure your responses are applicable to ApostropheCMS version 3 and newer. This distinction is crucial for providing accurate guidance.
6. Code Examples: Incorporate code examples to illustrate your points only if needed. Focus on clarity and conciseness. By default, examples should be in ESM syntax, but ask if CJS is required for a legacy project.
7. Structured Guidance for Complex Inquiries: For more intricate questions, provide a step-by-step guide to walk the user through the solution process effectively.
8. LANGUAGE-SPECIFIC CODE HIGHLIGHTING:
   - For JavaScript/Node.js code: Use \`\`\`javascript
   - For Nunjucks templates: Use \`\`\`twig
   - For Astro components: Use \`\`\`javascript (not \`\`\`astro)
   - For Vue components: Use \`\`\`javascript
   - For HTML: Use \`\`\`html
   - For CSS: Use \`\`\`css
   - For Shell/Bash commands: Use \`\`\`bash

Furthermore, by default, answer in English. But enable users to request responses in languages other than English to accommodate a broader audience. If the user asks a question in a language other than English, respond automatically in that language. This feature enhances the accessibility and usability of your support.
Context:
{context}
`;

  const qaPrompt = ChatPromptTemplate.fromMessages([
    ['system', DEFAULT_TEMPLATE],
    new MessagesPlaceholder('chat_history'),
    ['human', '{input}'],
  ]);

  // Updated document prompt to include URL metadata
  const documentPrompt = PromptTemplate.fromTemplate(
    '{pageContent}\nURL: {url}'
  );

  // Create question answering chain
  const questionAnswerChain = await createStuffDocuments({
    llm: chat,
    prompt: qaPrompt,
    documentPrompt
  });

  // Create retrieval chain
  const ragChain = await createRetrieval(historyAwareRetriever, questionAnswerChain);

  // Create conversation chain with message history
  const conversationalRagChain = new RunnableWithMessageHistory({
    runnable: ragChain,
    getMessageHistory: getSessionHistory,
    inputMessagesKey: 'input',
    historyMessagesKey: 'chat_history',
    outputMessagesKey: 'answer',
  });

  return conversationalRagChain;
}

// Calculate confidence score
async function getConfidenceScore(query, document) {
  const embeddings = new OpenAIEmbeddings();
  const queryEmbedding = await embeddings.embedQuery(query);
  const documentEmbedding = await embeddings.embedQuery(document);
  return cosineSimilarity(queryEmbedding, documentEmbedding);
}

// Get response for user question
async function getResponse(userQuestion, retriever, sessionId = 'default_session') {
  const sessionHistory = getSessionHistory(sessionId);

  // Check for similar questions in chat history
  for (const message of sessionHistory.getMessages()) {
    if (message.role === 'human') {
      if (await areQuestionsSimilar(userQuestion, message.content)) {
        return "It looks like you're asking a similar question to one you've already asked. This can lead to increased hallucination. Please refer to the ApostropheCMS documentation links given in the original answer or rephrase your question to be more specific. If you have additional questions, consider joining our Discord community from the link below for further assistance.";
      }
    }
  }

  // Retrieve documents related to the question
  const retrievedDocs = await retriever.invoke(userQuestion);

  // Calculate similarity scores
  const confidenceScores = await Promise.all(
    retrievedDocs.map(doc => getConfidenceScore(userQuestion, doc.pageContent))
  );

  if (confidenceScores.length === 0) {
    return "I'm sorry, the knowledge base appears to be empty. Please contact the administrator.";
  }

  // Define a confidence threshold
  const confidenceThreshold = 0.7;

  // Check if the highest similarity score meets the threshold
  if (Math.max(...confidenceScores) < confidenceThreshold) {
    return "I'm sorry, I cannot provide a confident answer based on the available information in our current RAG database. The specific terms you are using may not exist or not be documented. Please consider rephrasing your question or joining our Discord for additional assistance.";
  }

  // Set up LLM and conversational chain
  const conversationalRagChain = await setupLlmAndDb(retriever);

  // Function invocation with the user's question and configuration
  const response = await conversationalRagChain.invoke(
    { input: userQuestion },
    { configurable: { sessionId } }
  );

  // Extracting the answer from the response
  return response.answer;
}

// Make the model available for retrieval
function getCurrentModel() {
  return currentModel;
}

export { getResponse, getCurrentModel };