def word_break(s: str, word_dict: list[str]) -> bool:
    """Returns True if string s can be segmented into words from word_dict."""
    dp = [False] * (len(s) + 1)
    dp[0] = True
    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_dict:
                dp[i] = True
                break
    return dp[-1]