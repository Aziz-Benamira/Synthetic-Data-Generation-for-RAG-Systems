# reference prompts for the llm-as-judge functions
from utils import *

AnswerAccuracyJudge1Prompt = AnswerAccuracyPrompt(
    instruction = """You are a world class state of the art assistant for rating a User Answer given a Question. The Question is completely answered by the Reference Answer.
        Say 4, if User Answer is full contained and equivalent to Reference Answer in all terms, topics, numbers, metrics, dates and units.
        Say 2, if User Answer is partially contained and almost equivalent to Reference Answer in all terms, topics, numbers, metrics, dates and units.
        Say 0, if User Answer is not contained in Reference Answer or not accurate in all terms, topics, numbers, metrics, dates and units or the User Answer do not answer the question.
        Do not explain or justify your rating. Your rating must be only 4, 2 or 0 according to the instructions above.
        Return your response as JSON in this format: {"rating": X} where X is 0, 2, or 4.""",
    examples = [
        AnswerAccuracyExample(
            query="When was Albert Einstein born?",
            answer="Albert Einstein was born in 1879.",
            reference_answer="Albert Einstein was born on March 14, 1879.",
            output_score=2,
        ),
        AnswerAccuracyExample(
            query="What is the capital of France?",
            answer="Paris is the capital of France.",
            reference_answer="Paris is the capital of France.",
            output_score=4,
        ),
        AnswerAccuracyExample(
            query="What is the highest mountain?",
            answer="The Eiffel Tower is a famous landmark.",
            reference_answer="Mount Everest is the highest mountain.",
            output_score=0,
        ),
    ]
)

AnswerAccuracyJudge2Prompt = AnswerAccuracyPrompt(
    instruction = """I will rate the User Answer in comparison to the Reference Answer for a given Question.
        A rating of 4 indicates that the User Answer is entirely consistent with the Reference Answer, covering all aspects, topics, numbers, metrics, dates, and units.
        A rating of 2 signifies that the User Answer is mostly aligned with the Reference Answer, with minor discrepancies in some areas.
        A rating of 0 means that the User Answer is either inaccurate, incomplete, or unrelated to the Reference Answer, or it fails to address the Question.
        I will provide the rating without any explanation or justification, adhering to the following scale: 0 (no match), 2 (partial match), 4 (exact match).
        Do not explain or justify my rating. My rating must be only 4, 2 or 0 only.
        Return your response as JSON in this format: {"rating": X} where X is 0, 2, or 4.""",
    examples = [
        AnswerAccuracyExample(
            query="When was Albert Einstein born?",
            answer="Einstein was born in 1879 in Germany.",
            reference_answer="Albert Einstein was born on March 14, 1879 in Ulm, Germany.",
            output_score=2,
        ),
        AnswerAccuracyExample(
            query="What is the capital of France?",
            answer="The capital of France is Paris.",
            reference_answer="Paris is the capital of France.",
            output_score=4,
        ),
        AnswerAccuracyExample(
            query="What is the speed of light?",
            answer="The sun is a star.",
            reference_answer="The speed of light is approximately 299,792,458 meters per second.",
            output_score=0,
        ),
    ]  
)

ContextSupportPrompt = ContextSupportPrompt(
    instruction = """Evaluate if the following response is well-supported by the provided context.
        A score of 1.0 means the response is fully supported by the context with no hallucinations.
        A score of 0.5 means the response is partially supported, with some parts grounded in context and others questionable.
        A score of 0.0 means the response is either contradicted by the context or contains significant hallucinations not found in the context.
        Do not explain or justify your assessment.
        Return your response as JSON in this format: {"supported": X, "score": Y} where X is a boolean (true/false) and Y is 0.0, 0.5, or 1.0.""",
    examples = [
        ContextSupportExample(
            query="What is the capital of France?",
            answer="Paris is the capital of France.",
            context="France is a country in Western Europe. Paris is its largest city and serves as the capital.",
            supported=True,
            output_score=1.0,
        ),
        ContextSupportExample(
            query="What is the capital of France?",
            answer="Paris is the capital and Berlin is an important city.",
            context="France is a country in Western Europe. Paris is its capital.",
            supported=False,
            output_score=0.5,
        ),
        ContextSupportExample(
            query="What is the capital of France?",
            answer="The capital of France is London.",
            context="France is a country in Western Europe. Paris is its capital. London is the capital of the United Kingdom.",
            supported=False,
            output_score=0.0,
        ),
    ]
)

AnswerRelevancePrompt = AnswerRelevancePrompt(
    instruction = """Evaluate if the following response directly and completely answers the user's query.
        A score of 1.0 means the response fully answers the query with all necessary information.
        A score of 0.5 means the response partially answers the query but is incomplete or tangential.
        A score of 0.0 means the response does not answer the query at all or is completely irrelevant.
        Do not explain or justify your assessment.
        Return your response as JSON in this format: {"relevant": X, "score": Y} where X is a boolean (true/false) and Y is 0.0, 0.5, or 1.0.""",
    examples = [
        AnswerRelevanceExample(
            query="What is the capital of France?",
            answer="Paris is the capital of France.",
            relevant=True,
            output_score=1.0,
        ),
        AnswerRelevanceExample(
            query="What is the capital of France?",
            answer="Paris is a large city in France with many museums and landmarks.",
            relevant=True,
            output_score=0.5,
        ),
        AnswerRelevanceExample(
            query="What is the capital of France?",
            answer="France is a beautiful country in Europe.",
            relevant=False,
            output_score=0.0,
        ),
    ]
)

CoherencePrompt = CoherencePrompt(
    instruction = """Evaluate the coherence, clarity, and writing quality of the following text.
        A score of 1.0 means the text is clear, well-structured, and easy to understand.
        A score of 0.5 means the text is mostly coherent but has some unclear passages or awkward phrasing.
        A score of 0.0 means the text is incoherent, confusing, or difficult to understand.
        Do not explain or justify your assessment.
        Return your response as JSON in this format: {"coherent": X, "score": Y} where X is a boolean (true/false) and Y is 0.0, 0.5, or 1.0.""",
    examples = [
        CoherenceExample(
            answer="Paris is the capital of France. It is located in the north-central part of the country and is known for its art, fashion, and culture.",
            coherent=True,
            output_score=1.0,
        ),
        CoherenceExample(
            answer="Paris capital France located north is art culture. Fashion known is.",
            coherent=False,
            output_score=0.5,
        ),
        CoherenceExample(
            answer="xyzabc qwerty asdf lkjhg poiuy mnbvcx zxcvbn.",
            coherent=False,
            output_score=0.0,
        ),
    ]
)