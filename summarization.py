import json
import os

from langchain_together import ChatTogether
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

class Summarization:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("TOGETHER_API_KEY")
        self.llm = ChatTogether(model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")

        # Шаблон промта
        self.prompt_template = PromptTemplate(
            input_variables=["dialogue"],
            template="""
            Ты — AI-аналитик диалогов поддержки. Проанализируй диалог и верни JSON.

            Диалог:
            {dialogue}

            Формат вывода (только JSON!):
            {{
                "summary": "резюме диалога",
                "operator_performance": {{
                    "score": 1-5,
                    "reason": "обоснование"
                }},
                "crm_data": {{
                    "customer_issue": "проблема",
                    "issue_category": "connectivity/billing/technical/other",
                    "priority": "low/medium/high",
                    "solution_provided": "решение",
                    "effectiveness_score": 1-5,
                    "followup_required": true/false,
                    "customer_sentiment": "neutral/satisfied/frustrated/angry",
                    "key_phrases": ["список", "ключевых", "фраз"],
                    "internal_notes": "заметки",
                    "call_metrics": {{
                        "duration": 0,
                        "interruptions": 0
                    }}
                }}
            }}
            """
        )

        # Цепочка LangChain
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def extract_json(self, text: str) -> dict:
        """Извлекает JSON из текста."""
        start = text.find('{')
        end = text.rfind('}') + 1
        return json.loads(text[start:end])

    def analyze_dialogue(self, dialogue: str) -> dict:
        """Анализирует диалог и возвращает структурированные данные."""
        try:
            response = self.chain.run(dialogue=dialogue)
            return self.extract_json(response)
        except Exception as e:
            raise ValueError(f"Ошибка анализа: {e}")


# Пример использования
if __name__ == "__main__":
    example_dialogue = """
    Оператор: Здравствуйте! Чем могу помочь?
    Клиент: У меня не работает интернет уже час!
    Подсказка оператору: Сохраняйте спокойствие, уточните детали.
    Оператор: Понимаю ваше недовольство. Проверьте, пожалуйста, подключен ли кабель к роутеру.
    Клиент: Да, кабель на месте, но индикаторы не горят.
    Подсказка оператору: Предложите перезагрузку роутера.
    Оператор: Попробуйте перезагрузить роутер. Я пришлю вам инструкцию.
    Клиент: Спасибо, сейчас попробую.
    """

    agent = Summarization()

    try:
        result = agent.analyze_dialogue(example_dialogue)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Ошибка: {e}")