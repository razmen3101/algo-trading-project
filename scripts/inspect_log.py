p = r'C:\Users\User\AppData\Roaming\Code\User\workspaceStorage\dc07617f551a9add14f8dedaade2aad7\GitHub.copilot-chat\chat-session-resources\f841d3b3-e9ad-414c-a3c8-bdc1fd83d79b\call_15CHxOV8eBXzKXmavxKbQwcS__vscode-1780153286408\content.txt'
with open(p, 'r', encoding='utf-8') as f:
    s = f.read()

marker = '"A_RAW_with_return"'
mi = s.find(marker)
print('marker_index=', mi)
if mi != -1:
    start = max(0, mi-200)
    end = min(len(s), mi+2000)
    print(s[start:end])
else:
    print('marker not found')
