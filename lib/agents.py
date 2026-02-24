from typing import Dict, Any, Optional
from datetime import datetime
from lib.state_machine import StateMachine, AgentState, ConversationTurn
from lib.tools import GameRetrievalTool, RetrievalEvaluationTool, WebSearchTool
import json


class VideoGameAgent:
    
    def __init__(self, db_path: str = "./games_db"):
        self.state_machine = StateMachine()
        self.retrieval_tool = GameRetrievalTool(db_path=db_path)
        self.evaluation_tool = RetrievalEvaluationTool()
        self.web_search_tool = WebSearchTool()
        
        print(f"✓ Agent initialized - Session: {self.state_machine.session.session_id}")
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        session = self.state_machine.session
        turn_id = session.turn_count + 1
        
        turn = ConversationTurn(
            turn_id=turn_id,
            query=user_query,
            timestamp=datetime.now().isoformat(),
            agent_state=AgentState.IDLE,
            reasoning=""
        )
        
        reasoning_log = []
        
        reasoning_log.append("🔍 Step 1: Attempting retrieval from local game database...")
        self.state_machine.transition(AgentState.RETRIEVING)
        turn.agent_state = AgentState.RETRIEVING
        
        documents, distances = self.retrieval_tool.retrieve_game(user_query)
        reasoning_log.append(f"   Retrieved {len(documents)} candidates from database")
        
        turn.retrieval_results = {
            "documents": documents,
            "distances": distances,
            "num_results": len(documents)
        }
        
        reasoning_log.append("⚖️ Step 2: Evaluating retrieval confidence...")
        self.state_machine.transition(AgentState.EVALUATING)
        turn.agent_state = AgentState.EVALUATING
        
        evaluation = self.evaluation_tool.evaluate_retrieval(distances)
        confidence = evaluation["confidence"]
        score = evaluation["score"]
        
        reasoning_log.append(f"   Confidence: {confidence} (score: {score:.3f})")
        reasoning_log.append(f"   Rationale: {evaluation['rationale']}")
        
        turn.confidence = confidence
        
        use_local_result = confidence == "HIGH" and documents
        
        if use_local_result:
            reasoning_log.append("✓ High confidence in local results - Using database answer")
            self.state_machine.transition(AgentState.PROCESSING)
            turn.agent_state = AgentState.PROCESSING
            
            final_answer = documents[0]
            source = "Local Database"
            web_search_used = False
        else:
            reasoning_log.append("✗ Low/Medium confidence - Initiating web search fallback...")
            self.state_machine.transition(AgentState.SEARCHING_WEB)
            turn.agent_state = AgentState.SEARCHING_WEB
            
            reasoning_log.append("🌐 Step 3: Searching the web for information...")
            
            search_results = self.web_search_tool.game_web_search(user_query)
            
            if search_results["success"] and search_results["results"]:
                final_answer = search_results["results"][0]["content"]
                source = f"Web Search (Tavily) - {search_results['results'][0].get('title', 'Unknown')}"
                web_search_used = True
                reasoning_log.append(f"   Found web result: {search_results['results'][0]['title']}")
            else:
                final_answer = "No information found in local database or web search."
                source = "No Source"
                web_search_used = False
                reasoning_log.append("   ✗ No results found in web search")
            
            self.state_machine.transition(AgentState.PROCESSING)
            turn.agent_state = AgentState.PROCESSING
        
        self.state_machine.transition(AgentState.COMPLETE)
        turn.agent_state = AgentState.COMPLETE
        turn.final_answer = final_answer
        turn.source = source
        turn.web_search_used = web_search_used
        turn.reasoning = "\n".join(reasoning_log)
        
        self.state_machine.session.add_turn(turn)
        
        response = {
            "turn_id": turn_id,
            "query": user_query,
            "answer": final_answer,
            "confidence": confidence,
            "source": source,
            "web_search_used": web_search_used,
            "reasoning": turn.reasoning,
            "retrieval_details": {
                "num_candidates_found": len(documents),
                "best_score": score if distances else None
            }
        }
        
        self.state_machine.transition(AgentState.IDLE)
        
        return response
    
    def get_session_summary(self) -> Dict[str, Any]:
        session = self.state_machine.session
        
        return {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "turns_completed": session.turn_count,
            "conversation_history": [
                {
                    "turn_id": turn.turn_id,
                    "query": turn.query,
                    "source": turn.source,
                    "web_search_used": turn.web_search_used
                }
                for turn in session.conversation_history
            ]
        }
    
    def display_result(self, response: Dict[str, Any]) -> None:
        print("\n" + "="*80)
        print(f"TURN {response['turn_id']}: {response['query']}")
        print("="*80)
        
        print("\n📊 REASONING PROCESS:")
        print("-" * 80)
        print(response["reasoning"])
        
        print("\n📝 ANSWER:")
        print("-" * 80)
        print(response["answer"][:500] + "..." if len(response["answer"]) > 500 else response["answer"])
        
        print("\n📌 METADATA:")
        print("-" * 80)
        print(f"Confidence: {response['confidence']}")
        print(f"Source: {response['source']}")
        print(f"Web Search Used: {response['web_search_used']}")
        print(f"Candidates Found: {response['retrieval_details']['num_candidates_found']}")
        print("="*80 + "\n")