import os
from zhipuai import ZhipuAI
zhipu = os.getenv("ZhipuAI")
print(zhipu)
client = ZhipuAI(api_key=zhipu)
response = client.chat.completions.create(
    model="glm-4-alltools",  # Enter the model name you want to call
    messages=[
        {
            "role": "user",
            "content":[
                {
                    "type":"text",
                    "text":"Please help me query the national travel data for the Labor Day holiday from 2018 to 2024, and present the data trend in a bar chart."
                }
            ]
        }
    ],
    stream=True,
    tools=[
    {
        "type": "function",
        "function": {
            "name": "get_tourist_data_by_year",
            "description": " Used to query the national travel data for each year, input the year range (from_year, to_year), and return the corresponding travel data, including the total number of trips, the number of trips by different modes of transportation, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "description": "Mode of transportation, default is by_all, train = by_train, plane = by_plane, self-driving = by_car.",
                        "type": "string"
                    },
                    "from_year": {
                        "description": "Start year, formatted as yyyy.",
                        "type": "string"
                    },
                    "to_year": {
                        "description": "End year, formatted as yyyy.",
                        "type": "string"
                    }
                },
                "required": ["from_year","to_year"]
            }
        }
      },
      {
        "type": "code_interpreter"
      }
    ]
)

for chunk in response:
   print(chunk)