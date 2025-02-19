import requests



test_proxies = {
    "http": "127.0.0.1:1087",
    "https": "127.0.0.1:1087"
}

print("Testing proxy connection...")
print(f"Current proxy settings: {test_proxies}")

# 测试代理连接
test_response = requests.get(
    "https://www.google.com", 
    proxies=test_proxies, 
    timeout=5, 
    verify=False
)
print(test_response)
