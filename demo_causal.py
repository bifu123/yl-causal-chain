
## 关键字查找意志主体列表
def get_collapse(keyword:str, owner_id:str="222302526", url:str="http://192.168.66.39:8094/api/v1/causal/search/keyword", limit:int=10):
    import requests
    import json
    
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    from collapse_score_lite import OutcomeLabelEvaluator, get_best_outcome_label
    
    url = "http://192.168.66.39:8094/api/v1/causal/search/keyword"
    params = {
        "keyword": keyword,
        "owner_id": owner_id,
        "limit": limit
    }
    
    event_tuples = []
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        res_json = response.json()
        
        
        
        # 遍历列表提取 event_tuple 字段
        data_list = res_json.get("data", [])
        for item in data_list:
            event_tuple = item.get("event_tuple")
            event_tuples.append(event_tuple)
            
        
        # print(json.dumps(event_tuples, indent=4, ensure_ascii=False)) 
        
        return {"status": "seccuss", "event_tuples": get_best_outcome_label(event_tuples)}

    except Exception as e:
        return {"status": "fail", "err": f'{e}'}

        
if __name__ == "__main__":
    keyword="老李"
    r = get_collapse(keyword=keyword, limit=12)
    if r["status"] == "seccuss" and r["event_tuples"] != '':
        print(f'你当前对{keyword}的看法是:\n`{r["event_tuples"]}`\n**请基于你的看法决定回应措词**')
    else:
        print(f'你当前对{keyword}未坍塌任何果标签')
 
        
'''
.ylbot21) root@LXC-192-168-69-13-Ylbot2:~/ylbot21# python test.py
你当前对老李的看法是:
`老李不讲信用`
**请基于你的看法决定回应措词**
'''
