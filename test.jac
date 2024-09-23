import:py streamlit as st;
import:py requests;
import:py os;
import:py from langchain_community.document_loaders { PyPDFDirectoryLoader }
import:py from langchain_text_splitters { RecursiveCharacterTextSplitter }
import:py from langchain.schema.document { Document }
import:py from langchain_community.embeddings.ollama { OllamaEmbeddings }
import:py from langchain_community.vectorstores.chroma { Chroma }
import:py from mtllm.llms { Ollama }

obj RagEngine {
    has file_path: str = "docs";
    has chroma_path: str = "chroma";
    can postinit {
        documents: list = self.load_documents();
        chunks: list = self.split_documents(documents);
        self.add_to_chroma(chunks);
    }
    can load_documents {
        document_loader = PyPDFDirectoryLoader(self.file_path);
        return document_loader.load();
    }

    can split_documents(documents: list[Document]) {
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False
        );
        return text_splitter.split_documents(documents);
    }
    can get_embedding_function {
        embeddings = OllamaEmbeddings(model='nomic-embed-text');
        return embeddings;
    }
    can add_chunk_id(chunks: list[Document]) {
        last_page_id = None;
        current_chunk_index = 0;

        for chunk in chunks {
            source = chunk.metadata.get('source');
            page = chunk.metadata.get('page');
            current_page_id = f'{source}:{page}';

            if current_page_id == last_page_id {
                current_chunk_index += 1;
            } else {
                current_chunk_index = 0;
            }

            chunk_id = f'{current_page_id}:{current_chunk_index}';
            last_page_id = current_page_id;

            chunk.metadata['id'] = chunk_id;
        }

        return chunks;
    }
    can add_to_chroma(chunks: list[Document]) {
        db = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=self.get_embedding_function()
        );
        chunks_with_ids = self.add_chunk_id(chunks);

        existing_items = db.get();
        existing_ids = set(existing_items['ids']);

        new_chunks = [];
        for chunk in chunks_with_ids {
            if chunk.metadata['id'] not in existing_ids {
                new_chunks.append(chunk);
            }
        }

        if len(new_chunks) > 0 {
            print('adding new documents');
            new_chunk_ids = [chunk.metadata['id'] for chunk in new_chunks];
            db.add_documents(new_chunks, ids=new_chunk_ids);
        } else {
            print('no new documents to add');
        }
    }
    can get_from_chroma(query: str, chunk_nos: int = 5) {
        db = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=self.get_embedding_function()
        );
        results = db.similarity_search_with_score(query, k=chunk_nos);
        return results;
    }
}

glob llm = Ollama(model_name='llama3.1');
glob rag_engine: RagEngine = RagEngine();

walker init_router {
    can initialize_router {
        visit --> `?Router else {
                router_node = here --> Router();
                router_node --> RagChat();
                router_node --> QAChat();
                router_node --> ConsultChat();
                visit router_node;
            
        }
    }
}

walker infer {
    has message: str;
    has chat_history: list[dict];

    can init_router with `root {
        visit --> `?Router else {
            router_node = here --> Router();
            router_node --> RagChat();
            router_node --> QAChat();
            router_node --> ConsultChat();
            visit router_node;
        }
    }

    can route with Router {
        classification = here.classify(message = self.message);
        print(f"Routing to chat type: {classification}");
        visit --> `?Chat(?chat_type == classification);
    }
}

node Session {
    has id: str;
    has chat_history: list[dict];
    has status: int = 1;

    can llm_chat(
        message: str,
        chat_history: list[dict],
        agent_role: str,
        context: list
    ) -> str by llm();

    can chat with interact {
        self.chat_history.append({"role": "user", "content": here.message});
        data = rag_engine.get_from_chroma(query = here.message);
        response = here.llm_chat(
            message = here.message,
            chat_history = self.chat_history,
            agent_role = "You are a conversation agent designed to help users with their queries based on the documents provided",
            context = data
        );
        self.chat_history.append({"role": "assistant", "content": response});
        report { "response": response };
    }
}

node Router {
    can classify(message: str) -> ChatType by llm(method="Reason", temperature=0.0);
}

node Chat {
    has chat_type: ChatType;
}

node RagChat: Chat {
    has chat_type: ChatType = ChatType.RAG;

    can respond with infer {
        respond_with_llm(
            message: str,
            chat_history: list[dict],
            agent_role: str,
            context: list
        ) -> str by llm();

        data = rag_engine.get_from_chroma(query = here.message);
        here.response = respond_with_llm(
            message = here.message, 
            chat_history = here.chat_history, 
            agent_role = "You are a conversation agent designed to help users with their queries based on the documents provided", 
            context = data
        );
    }
}

node QAChat: Chat {
    has chat_type: ChatType = ChatType.QA;

    can respond with infer {
        respond_with_llm(
            message: str,
            chat_history: list[dict],
            agent_role: str
        ) -> str by llm();

        here.response = respond_with_llm(
            message = here.message,
            chat_history = here.chat_history, 
            agent_role = "You are a conversation agent designed to help users with their queries"
        );
    }
}

node ConsultChat: Chat {
    has chat_type: ChatType = ChatType.Consult;

    can respond with infer {
        respond_with_llm(
            message: str,
            chat_history: list[dict],
            agent_role: str
        ) -> str by llm();

        here.response = respond_with_llm(
            message = here.message, 
            chat_history = here.chat_history, 
            agent_role = "You are a conversation agent designed to provide expert advice and solutions across various domains for the task given by the users"
        );
    }
}

walker interact {
    has message: str;
    has session_id: str;

    can init_session with `root {
        visit --> `?Session(?id == self.session_id) else {
            session_node = here --> Session(id = self.session_id, chat_history = [], status = 1);
            print("Session Node Created");
            visit session_node;
        }
    }

    can chat with Session {
        here.chat_history.append({"role": "user", "content": self.message});
        response = infer(message = here.message, chat_history = here.chat_history) spawn `root;
        here.chat_history.append({"role": "assistant", "content": response.response});
        report { "response": response.response };
    }
}

# Forcefully invoke the router initialization at startup
init_router.initialize_router;