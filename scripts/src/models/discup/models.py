from transformers import AutoConfig

from transformers import GPT2LMHeadModel

def create_model(args):
    
    if args.model_name_or_path:
        
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        model = GPT2LMHeadModel.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config
        )
    else:
        print("Model path is not set!!!")        
        
    return model



def _create_model(model_path):
    if model_path:
        model = GPT2LMHeadModel.from_pretrained(model_path)
    else:
        print("Model path is not set!!!")        
        
    return model


def get_embedding_layer(args, model):

    embeddings = model.base_model.get_input_embeddings()

    return embeddings