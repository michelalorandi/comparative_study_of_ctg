import torch

from torch.utils.data import DataLoader

from src.models.discup.data import SentimentPrompt

import torch

from src.models.discup.prompt_tuning import Prompt_tuning
from src.models.discup.discriminator import PTuneForLAMA
from src.models.discup.distill_tuning import Distill_Tuning



class CTG(object):
    def __init__(self, args, prompts):
        self.args = args
        
        self.label_token ={
          "positive":'positive',
          "negative":'negative'
        }
    
        if  self.args.tuning_name == "prompt_tuning":
            self.model = Prompt_tuning(args, args.template, label_token = self.label_token)
            
        elif self.args.tuning_name == "distill_tuning":
            self.model = Distill_Tuning(args, args.template, label_token = self.label_token)
            
        elif self.args.tuning_name == "disc_tuning":
            self.label_token ={"positive":'good',"negative":'bad'}
            self.model = PTuneForLAMA(args, args.template, label_token = self.label_token)
        else:
            raise Exception("the tuning mode is not existing!")
        
        self.tokenizer = self.model.tokenizer
        
        # init the prompt encoder's parameters
        if args.embedding_checkpoint!= None:
            self.model.prompt_encoder.load_state_dict(self.load_prompt(args.embedding_checkpoint))
        
        print("sentiment control generation!")
        # data_path =  args.data_path
        pos_dataset = SentimentPrompt(tokenizer=self.tokenizer, prompts=prompts, max_length=self.args.max_prompt_length)
            
        self.pos_loader = DataLoader(pos_dataset, args.batch_size, num_workers=2,  shuffle=False)
        self.prompt_pad_length = self.args.prompt_pad_length
        
        self.generator_model = self.model.model        
        self.generateor_embedding = self.generator_model.get_input_embeddings()
        self.discrimirator_embedding = self.generateor_embedding
            
        
    def test(self):
        
        att = self.args.target_type
        desired_att_token = self.model.label_token_ids[att]
        
        final_texts = []
        for data in self.pos_loader:
            #print(data)
            #print('\n\n')
            x = data[0].squeeze(1).to(self.args.device)
            #print(x)
            #print('\n\n')
            musk = data[1].long().squeeze(1).to(self.args.device)
            desired_att = torch.tensor([desired_att_token]).expand(x.shape[0],-1).to(self.args.device)
                
            output_seq = self.model.generate(prompts_ids = x, max_length = self.args.max_length, desired_att=desired_att, beta = self.args.beta)
                
            text = self.tokenizer.batch_decode(output_seq["generated_tokens"], skip_special_tokens= True)
            text = [t.replace('\n', '') for t in text]
            final_texts.extend(text) 
        return final_texts
                         
    def load_prompt(self, embedding_checkpoint):
        checkpoint = torch.load(embedding_checkpoint)
        prompt_embedding = checkpoint['embedding']
        return prompt_embedding        
        

