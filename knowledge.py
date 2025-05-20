#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🤔 DeepKnowledge PDF & Web – Seu Assistente Avançado de Pesquisa!

Combine PDFs locais e URLs em um único CombinedKnowledgeBase, e faça
indexação apenas na primeira execução (recreate), economizando chamadas
à API nas execuções seguintes.

Dependências:
    pip install openai lancedb tantivy inquirer agno python-dotenv pandas

Defina sua chave da OpenAI no arquivo .env:
    OPENAI_API_KEY=sk-...
"""

import os
import sys
from textwrap import dedent
from typing import Optional, Tuple

# Verifica pandas (usado internamente pelo CombinedKnowledgeBase)
try:
    import pandas  # noqa: F401
except ImportError:
    print("ERROR: instale pandas (`pip install pandas`) para usar CombinedKnowledgeBase")
    sys.exit(1)

import typer
import inquirer
from dotenv import load_dotenv
from rich import print

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.embedder.openai import OpenAIEmbedder
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.knowledge.url import UrlKnowledge
from agno.knowledge.combined import CombinedKnowledgeBase
from agno.storage.agent.sqlite import SqliteAgentStorage

# ----------------------------------------------------------------------------
def initialize_knowledge_base() -> Tuple[CombinedKnowledgeBase, str]:
    """
    Configura LanceDb + PDFs locais + URLs. Não indexa de imediato.
    """
    index_dir = 'tmp/lancedb'
    vector_db = LanceDb(
        uri=index_dir,
        table_name='combined_documents',
        search_type=SearchType.hybrid,
        embedder=OpenAIEmbedder(id='text-embedding-3-small'),
    )

    # PDFKnowledgeBase usando o reader padrão (sem chunk=True)
    pdf_kb = PDFKnowledgeBase(
        path='docs',
        reader=PDFReader(),
        vector_db=vector_db,
    )

    url_kb = UrlKnowledge(
        urls=[
            'https://docs.agno.com/llms-full.txt',
            'https://www.anthropic.com/engineering/building-effective-agents',
            'https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview',
        ],
        vector_db=vector_db,
    )

    combined_kb = CombinedKnowledgeBase(
        sources=[pdf_kb, url_kb],
        vector_db=vector_db,
    )

    return combined_kb, index_dir

# ----------------------------------------------------------------------------
def get_agent_storage() -> SqliteAgentStorage:
    return SqliteAgentStorage(
        table_name='deep_knowledge_sessions',
        db_file='tmp/agents.db'
    )

# ----------------------------------------------------------------------------
def create_agent(session_id: Optional[str]) -> Agent:
    combined_kb, index_dir = initialize_knowledge_base()

    is_first_run = not (os.path.isdir(index_dir) and os.listdir(index_dir))
    if is_first_run:
        print('[bold yellow]Indexando base pela primeira vez (recreate=True)...[/bold yellow]')
        combined_kb.load(recreate=True)
    else:
        print('[bold green]Índice existente detectado. Carregando sem recriar (recreate=False).[/bold green]')
        combined_kb.load(recreate=False)

    storage = get_agent_storage()
    return Agent(
        name='DeepKnowledge',
        session_id=session_id,
        model=OpenAIChat(id='gpt-4o-mini'),
        description=dedent('''
            Você é o DeepKnowledge, um agente RAG que busca em PDFs locais
            e em URLs para responder perguntas de forma precisa.
        '''),
        instructions=dedent('''
            1. Analise e divida a pergunta.
            2. Identifique 3–5 termos de busca.
            3. Faça pelo menos 3 buscas na base.
            4. Cite sempre a fonte (PDF/página ou URL).
            5. Finalize somente quando cobrir todos os aspectos.
        '''),
        additional_context='Você tem acesso ao histórico de buscas desta sessão.',
        knowledge=combined_kb,    # passa o CombinedKnowledgeBase
        storage=storage,
        search_knowledge=True,   # Agentic RAG
        add_references=False,
        read_chat_history=True,
        show_tool_calls=True,
        markdown=True,
    )

# ----------------------------------------------------------------------------
def handle_session_selection() -> Optional[str]:
    storage = get_agent_storage()
    nova = typer.confirm('Deseja iniciar nova sessão?', default=False)
    if nova:
        return None

    sessions = storage.get_all_session_ids()
    if not sessions:
        print('Nenhuma sessão encontrada. Iniciando nova sessão.')
        return None

    print('\nSessões existentes:')
    for i, sid in enumerate(sessions, start=1):
        print(f'  {i}. {sid}')
    escolha = typer.prompt('Número da sessão (Enter para a mais recente)', default=1)
    try:
        return sessions[int(escolha) - 1]
    except Exception:
        return sessions[0]

# ----------------------------------------------------------------------------
def run_interactive_loop(agent: Agent):
    exemplos = [
        'O que são agentes de IA e como funcionam?',
        'Como funciona a indexação de documentos no Agno?',
        'Quais estratégias de prompt engineering são suportadas?',
        'Qual é o processo de recuperação de conhecimento no Agno?',
        'Como personalizar ferramentas no Agno?',
    ]

    while True:
        ops = [f'{i+1}. {q}' for i, q in enumerate(exemplos)] + ['Digitar pergunta...', 'Sair']
        ans = inquirer.prompt([inquirer.List('opt', message='Escolha:', choices=ops)])['opt']
        if ans == 'Sair':
            print('👋 Até a próxima!')
            break
        if ans == 'Digitar pergunta...':
            q = inquirer.prompt([inquirer.Text('custom', message='Digite sua pergunta:')])['custom']
        else:
            q = exemplos[int(ans.split('.')[0]) - 1]

        print(f'\n📝 Pergunta: {q}\n')
        agent.print_response(q, stream=True)
        print('\n' + '-'*80 + '\n')

# ----------------------------------------------------------------------------
def deep_knowledge_agent():
    load_dotenv()
    session_id = handle_session_selection()
    agent = create_agent(session_id)

    if session_id is None:
        print(f"[bold green]Nova sessão iniciada: {agent.session_id}[/bold green]\n")
    else:
        print(f"[bold blue]Continuando sessão: {session_id}[/bold blue]\n")

    run_interactive_loop(agent)

if __name__ == '__main__':
    typer.run(deep_knowledge_agent)
