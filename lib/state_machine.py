from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


class AgentState(Enum):
    IDLE = "idle"
    RETRIEVING = "retrieving"
    EVALUATING = "evaluating"
    SEARCHING_WEB = "searching_web"
    PROCESSING = "processing"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class ConversationTurn:
    turn_id: int
    query: str
    timestamp: str
    agent_state: AgentState
    retrieval_results: Optional[Dict[str, Any]] = None
    confidence: Optional[str] = None
    web_search_used: bool = False
    final_answer: Optional[str] = None
    source: Optional[str] = None
    reasoning: Optional[str] = None


@dataclass
class SessionState:
    session_id: str
    created_at: str
    current_state: AgentState = AgentState.IDLE
    conversation_history: List[ConversationTurn] = field(default_factory=list)
    turn_count: int = 0
    
    def add_turn(self, turn: ConversationTurn) -> None:
        self.conversation_history.append(turn)
        self.turn_count += 1
    
    def get_context(self) -> str:
        if not self.conversation_history:
            return "No previous conversation history."
        
        context = "Previous conversation turns:\n"
        for turn in self.conversation_history[-3:]:   
            context += f"\nTurn {turn.turn_id}: {turn.query}\n"
            if turn.final_answer:
                context += f"Answer: {turn.final_answer}\n"
        
        return context
    
    def transition_state(self, new_state: AgentState) -> None:
        self.current_state = new_state
    
    def reset(self) -> None:
        self.current_state = AgentState.IDLE


class StateMachine:    
    VALID_TRANSITIONS = {
        AgentState.IDLE: [AgentState.RETRIEVING],
        AgentState.RETRIEVING: [AgentState.EVALUATING, AgentState.ERROR],
        AgentState.EVALUATING: [AgentState.SEARCHING_WEB, AgentState.PROCESSING, AgentState.ERROR],
        AgentState.SEARCHING_WEB: [AgentState.PROCESSING, AgentState.ERROR],
        AgentState.PROCESSING: [AgentState.COMPLETE, AgentState.ERROR],
        AgentState.COMPLETE: [AgentState.IDLE],
        AgentState.ERROR: [AgentState.IDLE]
    }
    
    def __init__(self):
        self.session = SessionState(
            session_id=self._generate_session_id(),
            created_at=datetime.now().isoformat()
        )
    
    @staticmethod
    def _generate_session_id() -> str:
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def can_transition(self, from_state: AgentState, to_state: AgentState) -> bool:
        return to_state in self.VALID_TRANSITIONS.get(from_state, [])
    
    def transition(self, new_state: AgentState) -> bool:
        if self.can_transition(self.session.current_state, new_state):
            self.session.transition_state(new_state)
            return True
        return False
    
    def get_session_state(self) -> SessionState:
        return self.session
    
    def get_history(self) -> List[ConversationTurn]:
        return self.session.conversation_history
    
    def reset_session(self) -> None:
        self.session = SessionState(
            session_id=self._generate_session_id(),
            created_at=datetime.now().isoformat()
        )