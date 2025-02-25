from ragas import EvaluationDataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, ContextPrecision, AnswerRelevancy, ContextRecall, LLMContextPrecisionWithReference, LLMContextRecall
from langchain_google_genai import ChatGoogleGenerativeAI
from src.app.config.settings import settings
from langchain_groq import ChatGroq
import os
os.environ["RAGAS_APP_TOKEN"] = settings.RAGAS_APP_TOKEN

class RAGAsTest:
    def __init__(self) -> None:
        self.llm =  ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite-preview-02-05", api_key= settings.GEMINI_API_KEY)
        #self.llm = ChatGroq(model="llama-3.3-70b-versatile", api_key= settings.GROQ_API_KEY)
    
    def testing_loop(self, queries, relevant_doc, responses, references):
        
        dataset = []

        for query, relevant_doc ,response,reference in zip(queries,relevant_doc,responses,references):

            
            dataset.append(
                {
                    "user_input":query,
                    "retrieved_contexts":relevant_doc,
                    "response":response,
                    "reference":reference
                }
            )
            
            
        evaluation_dataset = EvaluationDataset.from_list(dataset)
        
        evaluator_llm = LangchainLLMWrapper(self.llm)
        result = evaluate(dataset=evaluation_dataset,metrics=[LLMContextPrecisionWithReference(), LLMContextRecall(), Faithfulness(), FactualCorrectness()],llm=evaluator_llm)
        print(f" FINAL Result : {result}")     
        
        result.upload()
        