import re, json, os

p = r'C:\Users\User\AppData\Roaming\Code\User\workspaceStorage\dc07617f551a9add14f8dedaade2aad7\GitHub.copilot-chat\chat-session-resources\f841d3b3-e9ad-414c-a3c8-bdc1fd83d79b\call_15CHxOV8eBXzKXmavxKbQwcS__vscode-1780153286408\content.txt'
with open(p, 'r', encoding='utf-8') as f:
    s = f.read()

# Find the JSON block starting near the experiments key
marker = '"A_RAW_with_return"'
mi = s.find(marker)
if mi == -1:
    raise SystemExit('marker not found')
# find the opening brace before marker
start = s.rfind('{', 0, mi)
if start == -1:
    raise SystemExit('opening brace not found')
# now find the matching closing brace by counting
depth = 0
end = None
for i in range(start, len(s)):
    if s[i] == '{':
        depth += 1
    elif s[i] == '}':
        depth -= 1
        if depth == 0:
            end = i
            break
if end is None:
    raise SystemExit('matching closing brace not found')
js = s[start:end+1]
obj = json.loads(js)
os.makedirs('outputs', exist_ok=True)
with open('outputs/residual_results.json','w',encoding='utf-8') as out:
    json.dump(obj, out, indent=2)
print('WROTE outputs/residual_results.json')
