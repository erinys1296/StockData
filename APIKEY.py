#無正式環境API
sim_api_key= "DTUFNoBDeCJuk5b2Vu6bUwtgpJKk8wYMHVH92eEz942x"     
sim_secret_key= "BcLZsDjCyrV2RZjZcNpWF1nMvgEYVPRW7dXva5pjgqqQ" 


#正式環境API
real_api_key = ""     #自行填入 
real_secret_key = ""     #自行填入 


def get_Key(Sim):
    if Sim == True:
        return sim_api_key
    elif Sim == False:
        return real_api_key
    else:
        print("The input of Sim should be True or False.")
        return None
    
def get_Secret(Sim):
    if Sim == True:
        return sim_secret_key
    elif Sim == False:
        return real_secret_key
    else:
        print("The input of Sim should be True or False.")
        return None
    
