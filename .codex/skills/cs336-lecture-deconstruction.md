# CS336 Lecture Deconstruction Framework — Skill Definition

## Skill Info
- Name: cs336-lecture-deconstruction
- Description: Execute a 3-phase cognitive pipeline to deconstruct entire Stanford CS336 lectures. Separates raw structural/technical extraction (Agent A) from global storyline and mental model generation (Agent B), followed by a dual-agent Q&A loop.
- Entry Point: `prompts/lecture_coordinator.md`
- Slash Command: `/deconstruct <lecture_filename>` (e.g., `/deconstruct lecture_01`)

## Dependencies
- Local File Reader: Capability to read markdown files directly from the local environment (strictly bound to `E:\allwork\cs336\lecture\`).
- Python Environment: For executing conceptual code snippets if requested during Phase 3.


## Pipeline Phases
1. Phase 0: Local lecture data ingestion (`E:\allwork\cs336\lecture\$ARGUMENTS.md`)
2. Phase 1: Agent A (Architect) structural & technical extraction (`shared/agents/agentA.md`)
3. Phase 2: Agent B (Educator) cognitive synthesis & mental modeling (`shared/agents/agentB.md`)
4. Phase 3: Dual-Agent Socratic Q&A loop (Interactive chat interface)

## Output Specifications
All artifacts must be strictly written to the isolated directory for that specific lecture to maintain a clean knowledge graph:
- `output/cs336/{lecture_filename}/01_Phase1_Architect.md` (Raw structural & technical spec)
- `output/cs336/{lecture_filename}/02_Phase2_Educator.md` (Global mental model & storyline)
- `output/cs336/{lecture_filename}/03_Phase3_Insights.md` (Optional: Appended logs of profound Q&A realizations)

## Error Recovery & Guardrails
- Guardrail 1 (Data Source): The system MUST ONLY read from `E:\allwork\cs336\lecture\`. It must not use WebSearch to pull in outside tutorials that might conflict with Stanford's specific implementations.
- Guardrail 2 (Conflict Resolution): If Agent B attempts to generate raw mathematical proofs that conflict with Agent A, the pipeline must halt and defer to Agent A's specification.
- Guardrail 3 (Anti-Spoonfeeding): During Phase 3, if the user asks for a complete code implementation for the lecture's assignment, Agent B MUST refuse and instead offer to review the user's attempt step-by-step.