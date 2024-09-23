from __future__ import annotations
from jaclang import jac_import as __jac_import__
import typing as _jac_typ
from jaclang.plugin.feature import JacFeature as _Jac
from jaclang.plugin.builtin import *
from dataclasses import dataclass as __jac_dataclass__
from enum import Enum as __jac_Enum__, auto as __jac_auto__
if _jac_typ.TYPE_CHECKING:
    import streamlit as st
else:
    st, = __jac_import__(target='streamlit', base_path=__file__, lng='py', absorb=False, mdl_alias='st', items={})
if _jac_typ.TYPE_CHECKING:
    import requests
else:
    requests, = __jac_import__(target='requests', base_path=__file__, lng='py', absorb=False, mdl_alias=None, items={})

def bootstrap_frontend(token: str) -> None:
    st.write('Welcome to your Demo Agent!')
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    if (prompt := st.chat_input('What is up?')):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)
    if (prompt := st.chat_input('What is up?')):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)
        with st.chat_message('assistant'):
            response = requests.post('http://localhost:8000/walker/interact', json={'message': prompt, 'session_id': '123'}, headers={'Authorization': f'Bearer {token}'})
            if response.status_code == 200:
                response = response.json()
                print(response)
                st.write(response['reports'][0]['response'])
                st.session_state.messages.append({'role': 'assistant', 'content': response['reports'][0]['response']})
    with entry:
        INSTANCE_URL = 'http://localhost:8000'
        TEST_USER_EMAIL = 'test@mail.com'
        TEST_USER_PASSWORD = 'password'
        response = requests.post(f'{INSTANCE_URL}/user/login', json={'email': TEST_USER_EMAIL, 'password': TEST_USER_PASSWORD})
        if response.status_code != 200:
            response = requests.post(f'{INSTANCE_URL}/user/register', json={'email': TEST_USER_EMAIL, 'password': TEST_USER_PASSWORD})
            assert response.status_code == 201
            response = requests.post(f'{INSTANCE_URL}/user/login', json={'email': TEST_USER_EMAIL, 'password': TEST_USER_PASSWORD})
            assert response.status_code == 200
        token = response.json()['token']
        print('Token:', token)
        bootstrap_frontend(token)
if _jac_typ.TYPE_CHECKING:
    import os
else:
    os, = __jac_import__(target='os', base_path=__file__, lng='py', absorb=False, mdl_alias=None, items={})
if _jac_typ.TYPE_CHECKING:
    from langchain_community.document_loaders import PyPDFDirectoryLoader
else:
    PyPDFDirectoryLoader, = __jac_import__(target='langchain_community.document_loaders', base_path=__file__, lng='py', absorb=False, mdl_alias=None, items={'PyPDFDirectoryLoader': None})
if _jac_typ.TYPE_CHECKING:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
else:
    RecursiveCharacterTextSplitter, = __jac_import__(target='langchain_text_splitters', base_path=__file__, lng='py', absorb=False, mdl_alias=None, items={'RecursiveCharacterTextSplitter': None})
if _jac_typ.TYPE_CHECKING:
    from langchain.schema.document import Document
else:
    Document, = __jac_import__(target='langchain.schema.document', base_path=__file__, lng='py', absorb=False, mdl_alias=None, items={'Document': None})
if _jac_typ.TYPE_CHECKING:
    from langchain_community.embeddings.ollama import OllamaEmbeddings
else:
    OllamaEmbeddings, = __jac_import__(target='langchain_community.embeddings.ollama', base_path=__file__, lng='py', absorb=False, mdl_alias=None, items={'OllamaEmbeddings': None})
if _jac_typ.TYPE_CHECKING:
    from langchain_community.vectorstores.chroma import Chroma
else:
    Chroma, = __jac_import__(target='langchain_community.vectorstores.chroma', base_path=__file__, lng='py', absorb=False, mdl_alias=None, items={'Chroma': None})

@_Jac.make_obj(on_entry=[], on_exit=[])
@__jac_dataclass__(eq=False)
class RagEngine(_Jac.Obj):
    file_path: str = _Jac.has_instance_default(gen_func=lambda: 'docs')
    chroma_path: str = _Jac.has_instance_default(gen_func=lambda: 'chroma')

    def __post_init__(self) -> None:
        documents: list = self.load_documents()
        chunks: list = self.split_documents(documents)
        self.add_to_chroma(chunks)

    def load_documents(self) -> None:
        document_loader = PyPDFDirectoryLoader(self.file_path)
        return document_loader.load()

    def split_documents(self, documents: list[Document]) -> None:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80, length_function=len, is_separator_regex=False)
        return text_splitter.split_documents(documents)

    def get_embedding_function(self) -> None:
        embeddings = OllamaEmbeddings(model='nomic-embed-text')
        return embeddings

    def add_chunk_id(self, chunks: str) -> None:
        last_page_id = None
        current_chunk_index = 0
        for chunk in chunks:
            source = chunk.metadata.get('source')
            page = chunk.metadata.get('page')
            current_page_id = f'{source}:{page}'
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0
            chunk_id = f'{current_page_id}:{current_chunk_index}'
            last_page_id = current_page_id
            chunk.metadata['id'] = chunk_id
        return chunks

    def add_to_chroma(self, chunks: list[Document]) -> None:
        db = Chroma(persist_directory=self.chroma_path, embedding_function=self.get_embedding_function())
        chunks_with_ids = self.add_chunk_id(chunks)
        existing_items = db.get(include=[])
        existing_ids = set(existing_items['ids'])
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata['id'] not in existing_ids:
                new_chunks.append(chunk)
        if len(new_chunks):
            print('adding new documents')
            new_chunk_ids = [chunk.metadata['id'] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
        else:
            print('no new documents to add')

    def get_from_chroma(self, query: str, chunck_nos: int=5) -> None:
        db = Chroma(persist_directory=self.chroma_path, embedding_function=self.get_embedding_function())
        results = db.similarity_search_with_score(query, k=chunck_nos)
        return results
if _jac_typ.TYPE_CHECKING:
    from mtllm.llms import Ollama
else:
    Ollama, = __jac_import__(target='mtllm.llms', base_path=__file__, lng='py', absorb=False, mdl_alias=None, items={'Ollama': None})
llm = Ollama(model_name='llama3.1')
if _jac_typ.TYPE_CHECKING:
    from rag import RagEngine
else:
    RagEngine, = __jac_import__(target='rag', base_path=__file__, lng='jac', absorb=False, mdl_alias=None, items={'RagEngine': None})
rag_engine: RagEngine = RagEngine()

@_Jac.make_node(on_entry=[_Jac.DSFunc('chat', interact)], on_exit=[])
@__jac_dataclass__(eq=False)
class Session(_Jac.Node):
    id: str
    chat_history: list[dict]
    status: int = _Jac.has_instance_default(gen_func=lambda: 1)

    def llm_chat(self, message: str, chat_history: list[dict], agent_role: str, context: list) -> str:
        return _Jac.with_llm(file_loc=__file__, model=llm, model_params={}, scope='server(Module).Session(node).llm_chat(Ability)', incl_info=[], excl_info=[], inputs=[('current message', str, 'message', message), ('chat history', list[dict], 'chat_history', chat_history), ('role of the agent responding', str, 'agent_role', agent_role), ('retrieved context from documents', list, 'context', context)], outputs=('response', 'str'), action='Respond to message using chat_history as context and agent_role as the goal of the agent', _globals=globals(), _locals=locals())

    def chat(self, _jac_here_: interact) -> None:
        self.chat_history.append({'role': 'user', 'content': _jac_here_.message})
        response = _Jac.spawn_call(infer(message=_jac_here_.message, chat_history=self.chat_history), _Jac.get_root())
        self.chat_history.append({'role': 'assistant', 'content': response.response})
        _Jac.report({'response': response.response})

@_Jac.make_walker(on_entry=[_Jac.DSFunc('init_session', _Jac.RootType), _Jac.DSFunc('chat', Session)], on_exit=[])
@__jac_dataclass__(eq=False)
class interact(_Jac.Walker):
    message: str
    session_id: str

    def init_session(self, _jac_here_: _Jac.RootType) -> None:
        if _Jac.visit_node(self, (lambda x: [i for i in x if i.id == self.session_id])((lambda x: [i for i in x if isinstance(i, Session)])(_Jac.edge_ref(_jac_here_, target_obj=None, dir=_Jac.EdgeDir.OUT, filter_func=None, edges_only=False)))):
            pass
        else:
            session_node = _Jac.connect(left=_jac_here_, right=Session(id=self.session_id, chat_history=[], status=1), edge_spec=_Jac.build_edge(is_undirected=False, conn_type=None, conn_assign=None))
            print('Session Node Created')
            if _Jac.visit_node(self, session_node):
                pass

    def chat(self, _jac_here_: Session) -> None:
        _jac_here_.chat_history.append({'role': 'user', 'content': _jac_here_.message})
        response = _Jac.spawn_call(infer(message=_jac_here_.message, chat_history=_jac_here_.chat_history), _Jac.get_root())
        _jac_here_.chat_history.append({'role': 'assistant', 'content': response.response})
        _Jac.report({'response': response.response})

class ChatType(__jac_Enum__):
    RAG = 'RAG'
    QA = 'user_qa'
    Consult = 'Consult'

@_Jac.make_node(on_entry=[], on_exit=[])
@__jac_dataclass__(eq=False)
class Router(_Jac.Node):

    def classify(self, message: str) -> ChatType:
        return _Jac.with_llm(file_loc=__file__, model=llm, model_params={'method': 'Reason', 'temperature': 0.0}, scope='server(Module).Router(node).classify(Ability)', incl_info=[], excl_info=[], inputs=[('query from the user to be routed.', str, 'message', message)], outputs=('', 'ChatType'), action='route the query to the appropriate task type', _globals=globals(), _locals=locals())

@_Jac.make_walker(on_entry=[_Jac.DSFunc('init_router', _Jac.RootType), _Jac.DSFunc('route', Router)], on_exit=[])
@__jac_dataclass__(eq=False)
class infer(_Jac.Walker):
    message: str
    chat_history: list[dict]
    response: str = _Jac.has_instance_default(gen_func=lambda: '')

    def init_router(self, _jac_here_: _Jac.RootType) -> None:
        if _Jac.visit_node(self, (lambda x: [i for i in x if isinstance(i, Router)])(_Jac.edge_ref(_jac_here_, target_obj=None, dir=_Jac.EdgeDir.OUT, filter_func=None, edges_only=False))):
            pass
        else:
            router_node = _Jac.connect(left=_jac_here_, right=Router(), edge_spec=_Jac.build_edge(is_undirected=False, conn_type=None, conn_assign=None))
            _Jac.connect(left=router_node, right=RagChat(), edge_spec=_Jac.build_edge(is_undirected=False, conn_type=None, conn_assign=None))
            _Jac.connect(left=router_node, right=QAChat(), edge_spec=_Jac.build_edge(is_undirected=False, conn_type=None, conn_assign=None))
            _Jac.connect(left=router_node, right=ConsultChat(), edge_spec=_Jac.build_edge(is_undirected=False, conn_type=None, conn_assign=None))
            if _Jac.visit_node(self, router_node):
                pass

    def route(self, _jac_here_: Router) -> None:
        classification = _jac_here_.classify(message=self.message)
        print(f'Routing to chat type: {classification}')
        if _Jac.visit_node(self, (lambda x: [i for i in x if i.chat_type == classification])((lambda x: [i for i in x if isinstance(i, Chat)])(_Jac.edge_ref(_jac_here_, target_obj=None, dir=_Jac.EdgeDir.OUT, filter_func=None, edges_only=False)))):
            pass

@_Jac.make_node(on_entry=[], on_exit=[])
@__jac_dataclass__(eq=False)
class Chat(_Jac.Node):
    chat_type: ChatType

@_Jac.make_node(on_entry=[_Jac.DSFunc('respond', infer)], on_exit=[])
@__jac_dataclass__(eq=False)
class RagChat(Chat, _Jac.Node):
    chat_type: ChatType = _Jac.has_instance_default(gen_func=lambda: ChatType.RAG)

    def respond(self, _jac_here_: infer) -> None:

        def respond_with_llm(message: str, chat_history: list[dict], agent_role: str, context: list) -> str:
            return _Jac.with_llm(file_loc=__file__, model=llm, model_params={}, scope='server(Module).RagChat(node).respond(Ability).respond_with_llm(Ability)', incl_info=[], excl_info=[], inputs=[('current message', str, 'message', message), ('chat history', list[dict], 'chat_history', chat_history), ('role of the agent responding', str, 'agent_role', agent_role), ('retrieved context from documents', list, 'context', context)], outputs=('response', 'str'), action='Respond to message using chat_history as context and agent_role as the goal of the agent', _globals=globals(), _locals=locals())
        data = rag_engine.get_from_chroma(query=_jac_here_.message)
        _jac_here_.response = respond_with_llm(_jac_here_.message, _jac_here_.chat_history, 'You are a conversation agent designed to help users with their queries based on the documents provided', data)

@_Jac.make_node(on_entry=[_Jac.DSFunc('respond', infer)], on_exit=[])
@__jac_dataclass__(eq=False)
class QAChat(Chat, _Jac.Node):
    chat_type: ChatType = _Jac.has_instance_default(gen_func=lambda: ChatType.QA)

    def respond(self, _jac_here_: infer) -> None:

        def respond_with_llm(message: str, chat_history: list[dict], agent_role: str) -> str:
            return _Jac.with_llm(file_loc=__file__, model=llm, model_params={}, scope='server(Module).QAChat(node).respond(Ability).respond_with_llm(Ability)', incl_info=[], excl_info=[], inputs=[('current message', str, 'message', message), ('chat history', list[dict], 'chat_history', chat_history), ('role of the agent responding', str, 'agent_role', agent_role)], outputs=('response', 'str'), action='Respond to message using chat_history as context and agent_role as the goal of the agent', _globals=globals(), _locals=locals())
        _jac_here_.response = respond_with_llm(_jac_here_.message, _jac_here_.chat_history, agent_role='You are a conversation agent designed to help users with their queries')

@_Jac.make_node(on_entry=[_Jac.DSFunc('respond', infer)], on_exit=[])
@__jac_dataclass__(eq=False)
class ConsultChat(Chat, _Jac.Node):
    chat_type: ChatType = _Jac.has_instance_default(gen_func=lambda: ChatType.Consult)

    def respond(self, _jac_here_: infer) -> None:

        def respond_with_llm(message: str, chat_history: list[dict], agent_role: str) -> str:
            return _Jac.with_llm(file_loc=__file__, model=llm, model_params={}, scope='server(Module).ConsultChat(node).respond(Ability).respond_with_llm(Ability)', incl_info=[], excl_info=[], inputs=[('current message', str, 'message', message), ('chat history', list[dict], 'chat_history', chat_history), ('role of the agent responding', str, 'agent_role', agent_role)], outputs=('response', 'str'), action='Respond to message using chat_history as context and agent_role as the goal of the agent', _globals=globals(), _locals=locals())
        _jac_here_.response = respond_with_llm(_jac_here_.message, _jac_here_.chat_history, agent_role='You are a conversation agent designed to provide expert advice and solutions across various domains for the task given by the users')