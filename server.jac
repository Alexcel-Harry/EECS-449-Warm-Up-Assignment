import:py streamlit as st;
import:py requests;
can bootstrap_frontend (token: str) {
    st.write("Welcome to your Demo Agent!");

    # Initialize chat history
    if "messages" not in st.session_state {
        st.session_state.messages = [];
    }
    for message in st.session_state.messages {
        with st.chat_message(message["role"]) {
            st.markdown(message["content"]);
        }
    }
    if prompt := st.chat_input("What is up?") {
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt});

        # Display user message in chat message container
        with st.chat_message("user") {
            st.markdown(prompt);
        }
    }
    if prompt := st.chat_input("What is up?") {
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt});

        # Display user message in chat message container
        with st.chat_message("user") {
            st.markdown(prompt);
        }

        # Display assistant response in chat message container
        with st.chat_message("assistant") {

            # Call walker API
            response = requests.post("http://localhost:8000/walker/interact", json={"message": prompt, "session_id": "123"},
                headers={"Authorization": f"Bearer {token}"}
            );

            if response.status_code == 200 {
                response = response.json();
                print(response);
                st.write(response["reports"][0]["response"]);

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response["reports"][0]["response"]});
            }
        }
    }
    with entry {

        INSTANCE_URL = "http://localhost:8000";
        TEST_USER_EMAIL = "test@mail.com";
        TEST_USER_PASSWORD = "password";

        response = requests.post(
            f"{INSTANCE_URL}/user/login",
            json={"email": TEST_USER_EMAIL, "password": TEST_USER_PASSWORD}
        );

        if response.status_code != 200 {
            # Try registering the user if login fails
            response = requests.post(
                f"{INSTANCE_URL}/user/register",
                json={
                    "email": TEST_USER_EMAIL,
                    "password": TEST_USER_PASSWORD
                }
            );
            assert response.status_code == 201;

            response = requests.post(
                f"{INSTANCE_URL}/user/login",
                json={"email": TEST_USER_EMAIL, "password": TEST_USER_PASSWORD}
            );
            assert response.status_code == 200;
        }

        token = response.json()["token"];

        print("Token:", token);

        bootstrap_frontend(token);
    }


}

import:py os;
import:py from langchain_community.document_loaders {PyPDFDirectoryLoader}
import:py from langchain_text_splitters {RecursiveCharacterTextSplitter}
import:py from langchain.schema.document {Document}
import:py from langchain_community.embeddings.ollama {OllamaEmbeddings}
import:py from langchain_community.vectorstores.chroma {Chroma}

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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False);
        return text_splitter.split_documents(documents);
    }
    can get_embedding_function {
        embeddings = OllamaEmbeddings(model='nomic-embed-text');
        return embeddings;
    }
    can add_chunk_id(chunks: str) {
        last_page_id = None;
        current_chunk_index = 0;

        for chunk in chunks {
            source = chunk.metadata.get('source');
            page = chunk.metadata.get('page');
            current_page_id = f'{source}:{page}';

            if current_page_id == last_page_id {
                current_chunk_index +=1;
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
        db = Chroma(persist_directory=self.chroma_path, embedding_function=self.get_embedding_function());
        chunks_with_ids = self.add_chunk_id(chunks);

        existing_items = db.get(include=[]);
        existing_ids = set(existing_items['ids']);

        new_chunks = [];
        for chunk in chunks_with_ids {
            if chunk.metadata['id'] not in existing_ids {
                new_chunks.append(chunk);
            }
        }

        if len(new_chunks) {
            print('adding new documents');
            new_chunk_ids = [chunk.metadata['id'] for chunk in new_chunks];
            db.add_documents(new_chunks, ids=new_chunk_ids);
        } else {
            print('no new documents to add');
        }
    }
    can get_from_chroma(query: str,chunck_nos: int=5) {
        db = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=self.get_embedding_function()
        );
        results = db.similarity_search_with_score(query,k=chunck_nos);
        return results;
    }
}

import:py from mtllm.llms {Ollama}

glob llm = Ollama(model_name='llama3.1');
# glob llm = Ollama(model_name='qwen2.5');
import:jac from rag {RagEngine}
glob rag_engine:RagEngine = RagEngine();


walker interact {
    has message: str;
    has session_id: str;

    can init_session with `root entry {
         visit [-->](`?Session)(?id == self.session_id) else {
            session_node = here ++> Session(id=self.session_id, chat_history=[], status=1);
            print("Session Node Created");

            visit session_node;
        }
    }
}


node Session {
    has id: str;
    has chat_history: list[dict];
    has status: int = 1;

    can 'Respond to message using chat_history as context and agent_role as the goal of the agent'
    llm_chat(
        message:'current message':str,
        chat_history: 'chat history':list[dict],
        agent_role:'role of the agent responding':str,
        context:'retrieved context from documents':list
    ) -> 'response':str by llm();
    can chat with interact entry {
        self.chat_history.append({"role": "user", "content": here.message});
        response = infer(message=here.message, chat_history=self.chat_history) spawn root;
        self.chat_history.append({"role": "assistant", "content": response.response});

        report {
            "response": response.response
        };
    }
}

enum ChatType {
    RAG : 'Need to use Retrievable information in specific documents to respond' = "RAG",
    QA : 'Given context is enough for an answer' = "user_qa",
    Consult : 'Given a task and need to provide expert advice and solutions across various domains' = "Consult"
}

node Router {
    can 'route the query to the appropriate task type'
    classify(message:'query from the user to be routed.':str) -> ChatType by llm(method="Reason", temperature=0.0);
}

walker infer {
    has message:str;
    has chat_history: list[dict];

    can init_router with `root entry {
        visit [-->](`?Router) else {
            router_node = here ++> Router();
            router_node ++> RagChat();
            router_node ++> QAChat();
            router_node ++> ConsultChat();
            visit router_node;
        }
    }
    can route with Router entry {
        classification = here.classify(message = self.message);
        print(f"Routing to chat type: {classification}");
        visit [-->](`?Chat)(?chat_type==classification);
    }
}
node Chat {
    has chat_type: ChatType;
}

node RagChat :Chat: {
    has chat_type: ChatType = ChatType.RAG;

    can respond with infer entry {
        can 'Respond to message using chat_history as context and agent_role as the goal of the agent'
        respond_with_llm(   message:'current message':str,
                    chat_history: 'chat history':list[dict],
                    agent_role:'role of the agent responding':str,
                    context:'retirved context from documents':list
                        ) -> 'response':str by llm();
        data = rag_engine.get_from_chroma(query=here.message);
        here.response = respond_with_llm(here.message, here.chat_history, "You are a conversation agent designed to help users with their queries based on the documents provided", data);
    }
}

node QAChat :Chat: {
    has chat_type: ChatType = ChatType.QA;

    can respond with infer entry {
        can 'Respond to message using chat_history as context and agent_role as the goal of the agent'
        respond_with_llm(   message:'current message':str,
            chat_history: 'chat history':list[dict],
            agent_role:'role of the agent responding':str
                ) -> 'response':str by llm();
        here.response = respond_with_llm(here.message, here.chat_history, agent_role="You are a conversation agent designed to help users with their queries");
    }
}

node ConsultChat :Chat: {
    has chat_type: ChatType = ChatType.Consult;

    can respond with infer entry {
        can 'Respond to message using chat_history as context and agent_role as the goal of the agent'
        respond_with_llm(   message:'current message':str,
            chat_history: 'chat history':list[dict],
            agent_role:'role of the agent responding':str
                ) -> 'response':str by llm();
        here.response = respond_with_llm(here.message, here.chat_history, agent_role="You are a conversation agent designed to provide expert advice and solutions across various domains for the task given by the users");
    }

}
