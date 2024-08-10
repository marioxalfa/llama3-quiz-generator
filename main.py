import transformers
import torch
from huggingface_hub import login

login(token="YOUR_TOKEN", add_to_git_credential=False)

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

prompt = """
#You are a sofisticated quiz generator.
- Generate quiz from the text delimited by 3 backticks.
- Return the numbers of questions that the user asks to you.

```
Nelle pagine che seguono dimostrerò che esiste una tecnica psicologica che rende possibile l'interpretazione dei sogni. Con l'applicazione di questa tecnica ogni sogno si rivelerà come una struttura psicologica, piena di significato. La concezione del sogno che era tenuta in età preistoriche dai popoli primitivi è un tema di così grande interesse che mi astengo dal trattarlo.
Gli antichi distinguevano tra i sogni veri e preziosi che venivano inviati al sognatore come avvertimenti o per predire eventi futuri, e i sogni vani, fraudolenti e vuoti il ​​cui scopo era di fuorviarlo o condurlo alla distruzione. "In generale, ci si aspettava che i sogni fornissero soluzioni importanti, ma non tutti i sogni venivano compresi immediatamente"
Fino a poco tempo fa la maggior parte degli autori era incline a trattare congiuntamente gli argomenti del sonno e dei sogni. In opere recenti c'è stata una tendenza a mantenersi più strettamente sul tema e a considerare, come un argomento speciale, i problemi separati della vita onirica. Ho avuto poche occasioni di occuparmi del problema del sonno, poiché si tratta essenzialmente di un problema fisiologico.
```

##Rules
- Return the quiz in this json format:
[
  {question: "question1", answers: {"answer1", "answer2", "answer3"}, correct_answer_index: 1},
  {question: "question2", answers: {"answer1", "answer2", "answer3"}, correct_answer_index: 2},
]
- Don't return anything else, just return the json!
- The json you return will be used in the frontend to render the quiz.

"""

messages = [
    {"role": "system", "content": prompt},
    {"role": "user", "content": "Generate 3 hard questions for my students"},
]

outputs = pipeline(
    messages,
    max_new_tokens=1024,
    do_sample=False,
    top_p=0.5,
    temperature=0.8
)
print(outputs[0]["generated_text"][-1])