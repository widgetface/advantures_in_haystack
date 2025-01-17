> ### Haystack is an open source framework for building production-ready LLM applications, retrieval-augmented generative pipelines and state-of-the-art search systems that work intelligently over large document collections. It lets you quickly try out the latest AI models while being flexible and easy to use.


## This is a repo for haystack related projects:

Note if you get an error
```
    RuntimeError: Failed to import transformers.generation.streamers because of the following error (look up to see its traceback):
    unsupported operand type(s) for |: 'type' and 'NoneType'
    
```
See the [issue](https://github.com/huggingface/transformers/issues/35639)

Solution: The cause here is the use of X | None types without a from __future__ import annotations line at the top of the file. 

.venv/lib/python3.9/site-packages/transformers/generation/streamers.py