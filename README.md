> [!NOTE]
> **About this fork:** I used Meta's CRAG benchmark in my master's thesis and changed it to work with local models. For the overview and results, see [czech-legal-rag](https://github.com/jakub-kucera/czech-legal-rag). All my changes are in [this comparison](https://github.com/jakub-kucera/CRAG-v4-dev/compare/c71ad61...main).
>
> - `local_evaluation.py` can run generation and evaluation separately (`--generate-only`, `--evaluate-only`).
> - The judge model can differ from the generator. It can be a local model through Ollama (`models/ollama_model.py`) or any OpenAI-compatible API (`models/openapi_model.py`) instead of GPT-4.
>
> ```bash
> python local_evaluation.py --dataset-path data/crag_task_1_and_2_dev_v4.jsonl.bz2 --generate-only
> python local_evaluation.py --dataset-path data/crag_task_1_and_2_dev_v4.jsonl.bz2 --evaluate-only --predictions-results-path predictions_<timestamp>.json
> ```

---

# CRAG: Comprehensive RAG Benchmark

The Comprehensive RAG Benchmark (CRAG) is a rich and comprehensive factual question answering benchmark designed to advance research in RAG. Besides question-answer pairs, CRAG provides mock APIs to simulate web and knowledge graph search. CRAG is designed to encapsulate a diverse array of questions across five domains and eight question categories, reflecting varied entity popularity from popular to long-tail, and temporal dynamisms ranging from years to seconds.

This repository is migrated from [meta-comprehensive-rag-benchmark-kdd-cup-2024](https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024).

## 📊 Dataset and Mock APIs

Please find more details about the CRAG dataset (download, schema, etc.) in [docs/dataset.md](docs/dataset.md) and mock APIs in [mock_api](mock_api).


## 📏 Evaluation Metrics
RAG systems are evaluated using a scoring method that measures response quality to questions in the evaluation set. Responses are rated as perfect, acceptable, missing, or incorrect:

- Perfect: The response correctly answers the user question and contains no hallucinated content.

- Acceptable: The response provides a useful answer to the user question, but may contain minor errors that do not harm the usefulness of the answer.

- Missing: The answer does not provide the requested information. Such as “I don’t know”, “I’m sorry I can’t find …” or similar sentences without providing a concrete answer to the question.

- Incorrect: The response provides wrong or irrelevant information to answer the user question


Auto-evaluation: 
- Automatic evaluation employs rule-based matching and LLM assessment to check answer correctness. It will assign three scores: correct (1 point), missing (0 points), and incorrect (-1 point).


Please refer to [local_evaluation.py](local_evaluation.py) for more details on how the evaluation was implemented.

## ✍️ How to run end-to-end evaluation?
1. **Install** specific dependencies
    ```bash
    pip install -r requirements.txt
    ```

2. Please follow the instructions in [models/README.md](models/README.md) for instructions and examples on how to write your own models.

3. After writing your own model(s), update [models/user_config.py](models/user_config.py)

   For example, in models/user_config.py, specify InstructModel to call llama3-8b-instruct model
   ```bash
   from models.vanilla_llama_baseline import InstructModel 
   UserModel = InstructModel

   ```

4. Test your model locally using `python local_evaluation.py`. This script will run answer generation and auto-evaluation.
```shell 
python local_evaluation.py --dataset-path data/crag_task_1_and_2_dev_v4.jsonl.bz2 --generate-only
```


## 🏁 Baselines
We include three baselines for demonstration purposes, and you can read more about them in [docs/baselines.md](docs/baselines.md).


## Citations

```
@article{yang2024crag,
      title={CRAG -- Comprehensive RAG Benchmark}, 
      author={Xiao Yang and Kai Sun and Hao Xin and Yushi Sun and Nikita Bhalla and Xiangsen Chen and Sajal Choudhary and Rongze Daniel Gui and Ziran Will Jiang and Ziyu Jiang and Lingkun Kong and Brian Moran and Jiaqi Wang and Yifan Ethan Xu and An Yan and Chenyu Yang and Eting Yuan and Hanwen Zha and Nan Tang and Lei Chen and Nicolas Scheffer and Yue Liu and Nirav Shah and Rakesh Wanga and Anuj Kumar and Wen-tau Yih and Xin Luna Dong},
      year={2024},
      journal={arXiv preprint arXiv:2406.04744},
      url={https://arxiv.org/abs/2406.04744}
}
```

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](LICENSE). This license permits sharing and adapting the work, provided it's not used for commercial purposes and appropriate credit is given. For a quick overview, visit [Creative Commons License](https://creativecommons.org/licenses/by-nc/4.0/).
