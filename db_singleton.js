// db_singleton.js
import { ChromaClient } from 'chromadb';
import { OpenAIEmbeddings } from '@langchain/openai';
import { Chroma } from '@langchain/community/vectorstores/chroma';
import dotenv from 'dotenv';
import fs from 'fs';
import path from 'path';

dotenv.config();

export class ChromaDB {
  static _instance = null;
  db = null;
  retriever = null;
  _embeddingCache = {}; // Cache for embeddings to reduce API calls

  static async getInstance() {
    if (!ChromaDB._instance) {
      ChromaDB._instance = new ChromaDB();
      await ChromaDB._instance._initialize();
    }
    return ChromaDB._instance;
  }

  async _initialize() {
    const maxRetries = 3;
    let retryCount = 0;

    while (retryCount < maxRetries) {
      try {
        // Initialize embeddings - use text-embedding-3-small to match original implementation
        const embeddings = new OpenAIEmbeddings({ 
          model: "text-embedding-3-small",
          disallowedSpecial: [] // equivalent to disallowed_special=()
        });

        // Define the path to your Chroma database
        const chromaDirectory = process.env.CHROMA_PATH || "./chroma_db/";

        // Ensure the directory exists
        if (!fs.existsSync(chromaDirectory)) {
          fs.mkdirSync(chromaDirectory, { recursive: true });
        }

        // Initialize Chroma with the embeddings
        this.db = await Chroma.fromExistingCollection(
          { collectionName: "langchain" },
          embeddings,
          { collectionMetadata: { "hnsw:space": "cosine" } },
          chromaDirectory
        );

        // Create a standard similarity retriever with parameters closer to DeepLake
        this.retriever = this.db.asRetriever({
          searchType: "similarity",
          k: 6
        });

        // Print Chroma collection stats
        try {
          console.log(`Chroma vectorstore successfully initialized at ${chromaDirectory}`);
          
          // Equivalent to db._collection.count() in Python
          const client = new ChromaClient();
          const collection = await client.getCollection({ name: "langchain" });
          const count = await collection.count();
          
          console.log(`Chroma collection count: ${count}`);
          console.log(`Chroma collection name: langchain`);
        } catch (e) {
          console.error(`Error getting Chroma stats: ${e}`);
        }

        return; // Successfully initialized, exit the retry loop
      } catch (e) {
        retryCount++;
        console.error(`Chroma initialization failed (attempt ${retryCount}/${maxRetries}): ${e}`);
        
        if (retryCount >= maxRetries) {
          console.error("All retries failed. Could not initialize ChromaDB.");
          throw e;
        }
        
        // Wait before retrying
        await new Promise(resolve => setTimeout(resolve, 2000));
      }
    }
  }

  async refreshConnection() {
    try {
      // Test connection by attempting to access the collection
      const client = new ChromaClient();
      const collection = await client.getCollection({ name: "langchain" });
      await collection.count();
    } catch (e) {
      console.log(`Refreshing Chroma connection due to: ${e}`);
      await this._initialize();
    }
  }

  async getCachedEmbedding(text, embeddingModel = null) {
    if (!this._embeddingCache[text]) {
      if (!embeddingModel) {
        embeddingModel = new OpenAIEmbeddings({ 
          model: "text-embedding-3-small",
          disallowedSpecial: []
        });
      }
      this._embeddingCache[text] = await embeddingModel.embedQuery(text);
    }
    return this._embeddingCache[text];
  }

  clearEmbeddingCache() {
    const cacheSize = Object.keys(this._embeddingCache).length;
    this._embeddingCache = {};
    console.log(`Cleared embedding cache (${cacheSize} entries)`);
  }

  fixDocumentMetadata(documents) {
    for (const doc of documents) {
      if (!doc.metadata) {
        doc.metadata = {};
      }

      // Ensure metadata is a dictionary
      if (typeof doc.metadata !== 'object') {
        doc.metadata = {};
      }

      // If URL is missing but can be found in the text, extract it and remove from content
      if (!doc.metadata.url && doc.pageContent && doc.pageContent.includes('Reference URL:')) {
        try {
          // Extract URL from text content
          const contentLines = doc.pageContent.split('\n');
          const urlLineIndices = contentLines
            .map((line, index) => ({ line, index }))
            .filter(item => item.line.includes('Reference URL:'))
            .map(item => item.index);

          if (urlLineIndices.length > 0) {
            const urlLineIndex = urlLineIndices[0];
            const url = contentLines[urlLineIndex].replace('Reference URL:', '').trim();

            // Store URL in metadata
            doc.metadata.url = url;

            // Remove the URL line from the content
            const newContentLines = contentLines.filter((_, i) => i !== urlLineIndex);
            doc.pageContent = newContentLines.join('\n');
            console.log(`Extracted URL to metadata: ${url}`);
          }
        } catch (e) {
          console.error(`Error extracting URL from content: ${e}`);
          console.error(e.stack); // More detailed error information
        }
      }

      // If still no URL, add a default one
      if (!doc.metadata.url) {
        doc.metadata.url = 'https://docs.apostrophecms.org/';
      }
    }

    return documents;
  }

  async healthCheck() {
    try {
      const client = new ChromaClient();
      const collection = await client.getCollection({ name: "langchain" });
      const count = await collection.count();
      return [true, `ChromaDB is healthy. Collection contains ${count} documents.`];
    } catch (e) {
      return [false, `ChromaDB is unhealthy: ${e.toString()}`];
    }
  }
}