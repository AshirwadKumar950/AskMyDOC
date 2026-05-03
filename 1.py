import google.generativeai as genai

genai.configure(api_key="AIzaSyAN4AyzpJmZ9WTzizi9sFtNVtrtxvBOCK8")

for model in genai.list_models():
    print(model.name)