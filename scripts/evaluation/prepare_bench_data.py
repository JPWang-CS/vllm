"""Create synthetic performance prompts with measured model chat lengths."""
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--tokens',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--kwargs',default='{}');a=p.parse_args()
    from transformers import AutoTokenizer
    t=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True,local_files_only=True)
    kwargs=json.loads(a.kwargs);ids=t.encode('data '*(a.tokens+128),add_special_tokens=False)
    def rendered(n):
        text=t.decode(ids[:n],skip_special_tokens=False)
        count=len(t.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,**kwargs))
        return text,count
    low,high=1,min(len(ids),a.tokens+64)
    while low<high:
        mid=(low+high+1)//2
        if rendered(mid)[1]<=a.tokens:low=mid
        else:high=mid-1
    text,count=rendered(low)
    if count>a.tokens:raise ValueError('chat template exceeds requested input length')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'prompt':text})+'\n')
    a.output.with_suffix('.metadata.json').write_text(json.dumps({'requested_chat_tokens':a.tokens,'actual_chat_tokens':count,
        'body_tokens':len(t.encode(text,add_special_tokens=False)),'chat_template_kwargs':kwargs},indent=2))


if __name__=='__main__':main()
