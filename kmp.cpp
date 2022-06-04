#include <iostream>
#include <string>
#include <vector>

using namespace std;

vector<int> build_next(const string& patt) {
  vector<int> next{0};  //第一个元素是0
  int prefix_len = 0;   //当前共同前后缀长度
  int i = 1;

  while (i < patt.size()) {
    // cout<<patt<<endl;
    if (patt[prefix_len] == patt[i]) {
      //根据上次的结果，如果当前进一位的值和之前的最长前缀的最后一个一样
      //最长前后缀+1，当前的next值就是最长前后缀长度
      prefix_len++;
      next.push_back(prefix_len);
      i++;
    } else {
      //如果不一样，最长前后缀就是前一个字符的值，就减少一个看看
      if (prefix_len - 1 < 0) {
        prefix_len = next[next.size() - prefix_len - 1];
      } else {
        prefix_len = next[prefix_len - 1];
      }
      if (prefix_len == 0) {
        next.push_back(0);
        i++;
      }
    }
  }

  return next;
}

int kmp_search(const string& string1, const string& patt) {
  vector<int> next = build_next(patt);  //获得需要匹配的字符串的next数组
  int i = 0;                            //主串中的指针
  int j = 0;                            //子串中的指针
  while (i < string1.size()) {
    if (string1[i] == patt[j]) {
      //字符串匹配，指针后移
      i++;
      j++;
    } else if (j > 0) {
      //字符匹配失败，子串中的指针根据next数组调整位置
      j = next[j - 1];
    } else {
      //子串的第一个字符就匹配失败
      i++;
    }

    if (j == patt.size()) {
      //匹配成功
      return i - j;
    }
  }

  return -1;
}

int main() {
  string a = "saadsaddfsdff";
  string b = "addf";
  int b_in_a_index = kmp_search(a, b);

  cout << b_in_a_index << endl;
  return 0;
}
