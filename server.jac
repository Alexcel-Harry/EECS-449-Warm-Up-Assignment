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

# glob llm = Ollama(model_name='llama3.1');
glob llm = Ollama(model_name='qwen2.5');
import:jac from rag {RagEngine}
glob rag_engine:RagEngine = RagEngine();
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
}
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
    can chat with Session entry {
        here.chat_history.append({"role": "user", "content": self.message});
        data = rag_engine.get_from_chroma(query=self.message);
        response = here.llm_chat(
            message=self.message,
            chat_history=here.chat_history,
            agent_role="You are a conversation agent designed to help users with their queries based on the documents provided",
            context=data
        );

        here.chat_history.append({"role": "assistant", "content": response});

        report {"response": response};
    }
}