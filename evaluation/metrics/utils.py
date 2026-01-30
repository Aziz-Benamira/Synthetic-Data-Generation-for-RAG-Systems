## TEMPLATE CLASSES

class BaseModelExample :
    """
    An example of an intended behavior for a model.
    """
    query = "Hello wassup",
    answer = "All good"

    def __init__(self, query, answer) -> None:
        self.query = query
        self.answer = answer

    def to_dict(self):
        """
        Docstring for to_text
        Generates a dict object, easy to dump in a json
        """
        return self.__dict__

class BasePrompt : 
    def __init__(self, instruction = None | str, examples = None | list[BaseModelExample]) -> None:
        if instruction == None:
            self.instruction = "You are a helpful agent. Help the user by answering the question"
        else:
            self.instruction = instruction
        if examples == None:
            self.examples = [BaseModelExample(query = "What's the capital of France",
                                 answer= "Paris")
            ]
        else:
            self.examples = examples
    def to_string(self):
        print("to_string doesn't work rn, i'm not sure what format to use")
        


## ANSWER ACCURACY CLASSES

class AnswerAccuracyExample(BaseModelExample) : 
    """
    Template for examples exchanges given to the answer accuracy function.
    """
    def __init__(self, query, answer, reference_answer, output_score) -> None:
        super().__init__(query, answer)
        self.reference = reference_answer
        self.output_score = output_score
    
class AnswerAccuracyPrompt(BasePrompt):
    
    def __init__(self, instruction = None, examples = None):
        super().__init__(instruction, examples)


## CONTEXT SUPPORT CLASSES

class ContextSupportExample(BaseModelExample):
    """
    Template for examples of context support evaluation.
    """
    def __init__(self, query, answer, context, supported, output_score) -> None:
        super().__init__(query, answer)
        self.context = context
        self.supported = supported
        self.output_score = output_score


class ContextSupportPrompt(BasePrompt):
    
    def __init__(self, instruction = None, examples = None):
        super().__init__(instruction, examples)


## ANSWER RELEVANCE CLASSES

class AnswerRelevanceExample(BaseModelExample):
    """
    Template for examples of answer relevance evaluation.
    """
    def __init__(self, query, answer, relevant, output_score) -> None:
        super().__init__(query, answer)
        self.relevant = relevant
        self.output_score = output_score


class AnswerRelevancePrompt(BasePrompt):
    
    def __init__(self, instruction = None, examples = None):
        super().__init__(instruction, examples)


## COHERENCE CLASSES

class CoherenceExample(BaseModelExample):
    """
    Template for examples of coherence evaluation.
    """
    def __init__(self, answer, coherent, output_score) -> None:
        self.answer = answer
        self.coherent = coherent
        self.output_score = output_score


class CoherencePrompt(BasePrompt):
    
    def __init__(self, instruction = None, examples = None):
        super().__init__(instruction, examples)