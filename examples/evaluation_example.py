#!/usr/bin/env python3
"""Evaluation example for RL-KG-Agent."""

import json
from pathlib import Path

# Add src to path for development
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_kg_agent.knowledge.kg_loader import KnowledgeGraphLoader, SPARQLQueryGenerator
from rl_kg_agent.knowledge.internal_kg import InternalKnowledgeGraph
from rl_kg_agent.actions.action_manager import ActionManager
from rl_kg_agent.agents.ppo_agent import PPOKGAgent
from rl_kg_agent.utils.reward_calculator import RewardCalculator
from rl_kg_agent.utils.llm_client import LLMClient


def main():
    """Evaluation example."""
    print("📊 RL-KG-Agent Evaluation Example")

    # Set up paths
    examples_dir = Path(__file__).parent
    ttl_file = examples_dir / "sample_knowledge_graph.ttl"
    test_questions_file = examples_dir / "evaluation_questions.json"
    model_path = "demo_trained_model"

    if not ttl_file.exists():
        print(f"❌ TTL file not found: {ttl_file}")
        return

    try:
        # 1. Load Test Questions
        print(f"📋 Loading test questions from {test_questions_file}")
        if test_questions_file.exists():
            with open(test_questions_file, 'r') as f:
                test_questions = json.load(f)
        else:
            print("⚠️  Test questions file not found, using built-in examples")
            test_questions = [
                {
                    "question": "What is the capital of France?",
                    "expected_answer": "Paris",
                    "question_type": "factual",
                    "difficulty": "easy"
                },
                {
                    "question": "Who wrote Romeo and Juliet?",
                    "expected_answer": "William Shakespeare",
                    "question_type": "factual",
                    "difficulty": "easy"
                },
                {
                    "question": "How does photosynthesis work?",
                    "expected_answer": "Plants convert light energy into chemical energy using chlorophyll",
                    "question_type": "explanation",
                    "difficulty": "medium"
                }
            ]

        print(f"   Loaded {len(test_questions)} test questions")

        # 2. Initialize Components
        print("📚 Initializing components...")

        kg_loader = KnowledgeGraphLoader(str(ttl_file))
        sparql_generator = SPARQLQueryGenerator()
        internal_kg = InternalKnowledgeGraph("eval_internal_kg.pkl")
        llm_client = LLMClient()

        action_manager = ActionManager(kg_loader, sparql_generator, internal_kg, llm_client)
        reward_calculator = RewardCalculator()

        # 3. Initialize Agent
        print("🤖 Initializing agent...")
        agent = PPOKGAgent(action_manager, reward_calculator, internal_kg)

        # Try to load trained model
        if Path(model_path).exists():
            try:
                print(f"🧠 Loading trained model from {model_path}")
                agent.load_model(model_path)
                using_trained_model = True
            except Exception as e:
                print(f"⚠️  Failed to load model: {e}")
                print("   Using untrained agent")
                using_trained_model = False
        else:
            print(f"⚠️  Model not found at {model_path}, using untrained agent")
            using_trained_model = False

        # 4. Run Evaluation
        print("\n🔍 Running evaluation...")

        results = []
        total_reward = 0
        correct_answers = 0

        for i, test_case in enumerate(test_questions):
            question = test_case["question"]
            expected = test_case["expected_answer"]
            difficulty = test_case.get("difficulty", "unknown")

            print(f"\n📝 [{i+1}/{len(test_questions)}] {question}")
            print(f"   Expected: {expected}")
            print(f"   Difficulty: {difficulty}")

            # Extract simple entities
            entities = _extract_simple_entities(question)

            if using_trained_model:
                # Use trained agent
                try:
                    result = agent.execute_query(question, entities, expected, max_steps=5)

                    reward = result["total_reward"]
                    total_reward += reward

                    # Extract final response
                    final_response = ""
                    if result["steps_taken"]:
                        last_step = result["steps_taken"][-1]
                        final_response = last_step.get("info", {}).get("response", "No response")

                    # Simple correctness check
                    is_correct = _check_answer_correctness(final_response, expected)
                    if is_correct:
                        correct_answers += 1

                    print(f"   🤖 Agent Response: {final_response}")
                    print(f"   📊 Reward: {reward:.3f}")
                    print(f"   ✅ Correct: {is_correct}")

                    # Store detailed result
                    results.append({
                        "question": question,
                        "expected_answer": expected,
                        "agent_response": final_response,
                        "reward": reward,
                        "correct": is_correct,
                        "difficulty": difficulty,
                        "steps_taken": len(result["steps_taken"])
                    })

                except Exception as e:
                    print(f"   ❌ Evaluation failed: {e}")
                    results.append({
                        "question": question,
                        "expected_answer": expected,
                        "agent_response": f"Error: {e}",
                        "reward": 0.0,
                        "correct": False,
                        "difficulty": difficulty,
                        "steps_taken": 0
                    })

            else:
                # Use action manager directly
                context = {
                    "query": question,
                    "entities": entities,
                    "expected_answer": expected,
                    "history": []
                }

                context = action_manager.update_context_with_internal_knowledge(context)
                recommendations = action_manager.get_action_recommendations(context)

                if recommendations:
                    best_action, confidence = recommendations[0]
                    result = action_manager.execute_action(best_action, context)

                    if result.success:
                        reward_components = reward_calculator.calculate_reward(context, result, expected)
                        reward = reward_components.total
                        total_reward += reward

                        is_correct = _check_answer_correctness(result.response, expected)
                        if is_correct:
                            correct_answers += 1

                        print(f"   🤖 Response: {result.response}")
                        print(f"   🎯 Action: {best_action.name}")
                        print(f"   📊 Reward: {reward:.3f}")
                        print(f"   ✅ Correct: {is_correct}")

                        results.append({
                            "question": question,
                            "expected_answer": expected,
                            "agent_response": result.response,
                            "reward": reward,
                            "correct": is_correct,
                            "difficulty": difficulty,
                            "action_taken": best_action.name
                        })
                    else:
                        print(f"   ❌ Action failed: {result.response}")
                        results.append({
                            "question": question,
                            "expected_answer": expected,
                            "agent_response": result.response,
                            "reward": 0.0,
                            "correct": False,
                            "difficulty": difficulty,
                            "action_taken": best_action.name
                        })

        # 5. Calculate and Display Metrics
        print(f"\n📈 Evaluation Results Summary:")
        print(f"   Total Questions: {len(test_questions)}")
        print(f"   Correct Answers: {correct_answers}/{len(test_questions)} ({correct_answers/len(test_questions)*100:.1f}%)")
        print(f"   Average Reward: {total_reward/len(test_questions):.3f}")

        # Break down by difficulty
        difficulty_stats = {}
        for result in results:
            diff = result["difficulty"]
            if diff not in difficulty_stats:
                difficulty_stats[diff] = {"correct": 0, "total": 0, "total_reward": 0}

            difficulty_stats[diff]["total"] += 1
            if result["correct"]:
                difficulty_stats[diff]["correct"] += 1
            difficulty_stats[diff]["total_reward"] += result["reward"]

        print(f"\n📊 Performance by Difficulty:")
        for diff, stats in difficulty_stats.items():
            accuracy = stats["correct"] / stats["total"] * 100
            avg_reward = stats["total_reward"] / stats["total"]
            print(f"   {diff.capitalize()}: {stats['correct']}/{stats['total']} ({accuracy:.1f}%) - Avg Reward: {avg_reward:.3f}")

        # 6. Save Results
        results_file = examples_dir / "evaluation_results.json"
        print(f"\n💾 Saving results to {results_file}")

        summary = {
            "summary": {
                "total_questions": len(test_questions),
                "correct_answers": correct_answers,
                "accuracy": correct_answers / len(test_questions),
                "average_reward": total_reward / len(test_questions),
                "using_trained_model": using_trained_model
            },
            "difficulty_breakdown": difficulty_stats,
            "detailed_results": results
        }

        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print("✅ Evaluation completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def _extract_simple_entities(text):
    """Simple entity extraction for demo."""
    words = text.split()
    entities = [word.rstrip('?.,!') for word in words if word[0].isupper() and len(word) > 2]
    return entities[:5]


def _check_answer_correctness(response, expected):
    """Simple answer correctness check."""
    # Very basic check - look for expected answer in response
    return expected.lower() in response.lower()


if __name__ == "__main__":
    main()