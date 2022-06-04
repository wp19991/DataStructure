# 基本方法
## vector 
- `#include <vector>`
- 新建一个空的vector
    - `vector<int> res;`
- 初始化一个二维的matrix
    - 行M,列N,且值自定义为data;
    - `vector<vector<int>> matrix(M,vector<int>(N,data));`
    - `vector<vector<int>> matrix2{ {1},{1,1} };//学会用大括号初始化二维数组`
- 装入元素
    - `res.push_back(12);`
- 装入另一个vector
    - `res.insert(res.end(),temp.begin(),temp.end());`

- 弹出容器中最后一个元素(容器必须非空)
    - `res.pop_back(); // 弹出容器中最后一个元素(容器必须非空)`
- 清空容器
    - `res.clear(); // 清空容器,相当于调用erase(begin(), end())`
- 获得元素个数
    - `int len = res.size();`
- vector反转
    - `reverse(tmp.begin(), tmp.end());`

## stack
- `#include <stack>`
- 新建一个栈
    - `stack<TreeNode *> st;`
- 加入元素,将 root 置于栈顶
    - `st.push(root);`
- 获得栈顶元素
    - `TreeNode * temp = st.top();`
- 获得元素个数
    - `int len = st.size();`
- 栈是否为空
    - `st.empty();//栈空返回 true 否则 false`
- 删除栈顶元素
    - `st.pop();`

## queue
- `#include <queue>`
- 新建一个队列
    - `queue<TreeNode *> qe;`
- 获得元素个数
    - `int len = qe.size();`
- 队列是否为空
    - `qe.empty();//队列空返回 true 否则 false`
- 取队头元素
    - `qe.front();`
- 队头元素出队
    - `qe.pop();`

```cpp
struct cmp{
    operator ()(const Node &q,const Node &p){
        return q.var<p.var;
    }
};

int main(int argc, char *argv[]){
    //优先队列的实质还是堆排序
    //默认的优先队列是以大顶堆构造，即序列中最大数的优先级最高
    //这里自定义构造比较函数，可以使小的具有优先级
    priority_queue<Node,vector<Node>,cmp >q;
 
    for(int i=10;i>0;i--){
        q.push(i);
    }
 
    while(!q.empty()){
        cout<<q.top().name<<" ";
        q.pop();
    }//1,2,3...10
 
    return 0;
}
```

## unordered_set
- `#include <unordered_set>`
- 创建一个存int类型的unordered_set
    - unordered_set<int> myset;
    - unordered_set<string> things {16}; // 可以存16个元素
    - unordered_set<string> words {"one", "two", "three", "four"};// 存入这个string的元素
- myset.insert(12) //增
    - ret = myset.insert(1);
    - 如果插入的时候有重复的，它会返回false
- myset.erase(12) //删
- myset.find(12) //查

## unordered_map
- `#include <unordered_map>`
- 创建一个key为string类型,value为int类型的unordered_map
    - `unordered_map<string, int> unomap;`
- 使用变量方式,插入一个元素
    - `string key="k1"; int value=4;`
    - `unomap.emplace(key, value);`
- 也可以直接写上key和value的值
    - `unomap.emplace("k2", 7);`
- 使用pair插入键值对
    - `unomap.insert(pair<string, int>("k3", 3));`
- 通过key值来访问value
    - `cout<<unomap["k1"];`
- 遍历整个map,输出key及其对应的value值
    - `for(auto x:unomap) cout<<x.first<<"  "<<x.second<<endl;`
- 遍历整个map,并根据其key值,查看对应的value值
    - `for(auto x:unomap) cout<<unomap[x.first]<<endl;`


# 题目
## kmp
```cpp
#include <iostream>
#include <string>
#include <vector>

using namespace std;

vector<int> build_next(const string &patt) {
    vector<int> next{0};//第一个元素是0
    int prefix_len = 0;//当前共同前后缀长度
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

int kmp_search(const string &string1, const string &patt) {
    vector<int> next = build_next(patt);//获得需要匹配的字符串的next数组
    int i = 0;//主串中的指针
    int j = 0;//子串中的指针
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
```

## 链表相关
### 反转链表
- 输入: `1->2->3->4->5->NULL`
- 输出: `5->4->3->2->1->NULL`
```cpp
#include <iostream>
using namespace std;
class Solution {
public:
    ListNode *reverseList(ListNode *head) {
        ListNode *prev = NULL;
        ListNode *curr = head;
        while (curr) {
            ListNode *next = curr->next;
            curr->next = prev;
            prev = curr;
            curr = next;
        }//while
        return prev;
    }
};
```

### 复杂链表的复制
- 请实现 copyRandomList 函数,复制一个复杂链表.
- 在复杂链表中,每个节点除了有一个 next 指针指向下一个节点,还有一个 random 指针指向链表中的任意节点或者 null.
```cpp
#include <iostream>
#include <unordered_map>
using namespace std;
class Solution {
public:
    unordered_map<Node*,Node*> cachedNode;
    Node *copyRandomList(Node *head) {
        if (head==NULL) {return NULL;}

        if (!cachedNode.count(head)) {
            Node *headNew = new Node(head->val);
            cachedNode[head] = headNew;
            headNew->next = copyRandomList(head->next);
            headNew->random = copyRandomList(head->random);
        }//if
        return cachedNode[head];
    }
};
```

## 数组类似于顺序表的应用
### 增减字符串匹配
- 给定只含 "I"（增大）或 "D"（减小）的字符串 S ，令 N = S.length。
- 返回 [0, 1, ..., N] 的任意排列 A 使得对于所有 i = 0, ..., N-1，都有：
    - 如果 S[i] == "I"，那么 A[i] < A[i+1]
    - 如果 S[i] == "D"，那么 A[i] > A[i+1]
- 示例 1：
    - 输入："IDID"
    - 输出：[0,4,1,3,2]
- 示例 2：
    - 输入："III"
    - 输出：[0,1,2,3]
- 示例 3：
    - 输入："DDI"
    - 输出：[3,2,0,1]
```cpp
class Solution {
public:
    //设numi=0,numd=S.length()，遇到D就把numd--推后，遇到I就把numi++推后。
    vector<int> diStringMatch(string S) {
        int len=S.length()
        int numi=0
        int numd=len;
        vector<int> res;
        for(int i=0;i<len;i++){
            if(S[i]=='I'){
                res.push_back(numi++);
            }
            else if(S[i]=='D'){
                res.push_back(numd--);
            }
        }
        res.push_back(numd);
        return res;
    }
};
```

### 二维数组中的查找
- 在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。
- 请完成一个高效的函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
```cpp
#include <iostream>
#include <vector>
using namespace std;
class Solution {
public:
    bool findNumberIn2DArray(vector<vector<int>>& matrix, int target) {
        //从右上角开始走，利用这个顺序关系可以在O(m+n)的复杂度下解决这个题：
        //如果当前位置元素比target小，则row++
        //如果当前位置元素比target大，则col--
        //如果相等，返回true
        //如果越界了还没找到，说明不存在，返回false
        if(matrix.size()==0)return false;
        int m=matrix.size()
        int n=matrix[0].size();
        int row=0,col=n-1;
        while(row<m&&col>=0){
            if(matrix[row][col]>target){
                col--;
            }else if(matrix[row][col]<target){
                row++;
            }else{
                return true;
            }
        }
        return false;
    }
};
```
## 串
### 把数字翻译成字符串
- 给定一个数字，我们按照如下规则把它翻译为字符串：
- 0 翻译成 a，1 翻译成 b，……，11 翻译成 l，……，25 翻译成 z。
- 一个数字可能有多个翻译。
- 例如 12258 有 5 种不同的翻译，它们分别是 bccfi、bwfi、bczi、mcfi 和 mzi。
- 请编程实现一个函数用来计算一个数字有多少种不同的翻译方法。
```cpp
/*
DP O(N)O(N)
还记得经典的爬楼梯(斐波那契数列)吗？每次可以走1步或者2步，问n个台阶一共有多少种爬楼梯的方法？
dp[i]=dp[i-1]+dp[i-2]

这道题相当于加了一些限制条件。

这个题可以正推或者倒推，我采用的方法是倒着推
以dp[i]表示从字符串i位开始到末尾，最大的翻译次数。

dp[i] = dp[i+1] // default, 比如都是67876878，这种只有1种解码方式，不会增加 = dp[i+1] + dp[i+2] // when s[i]=='1'||(s[i]=='2'&&s[i+1]<'6') 这种情况的出现会使解码次数增加
举个例子12xxxxxx;将1作为单独的一个数看，解码方法和2xxxxxx相同
；将12作为一个整体看，解码方法数量和xxxxxx相同。
最终的数量是二者之和。
*/
class Solution {
public:
    int getTranslationCount(string s) {
        int n = s.size();
        if(!n) return 0;
        if(n==1) return 1;

        vector<int> dp(n+1, 0);
        dp[n-1] = 1;
        for(int i=n-2;i>=0;i--){
            dp[i] = dp[i+1];
            if(s[i]=='1' || (s[i]=='2' && s[i+1]<'6')){
                dp[i] += dp[i+2];
            }
        }
        return dp[0];
    }
};
```

## 栈与队列应用
### 用两个栈实现队列
```cpp
#include <iostream>
#include <stack>
using namespace std;
class Solution {
public:
    stack<int> sck1;
    stack<int> sck2;
    // 在队列尾部插入整数
    void appendTail(int value) {
        sck1.push(value);
    }
    // 在队列头部删除整数
    int deleteHead() {
        if (sck2.empty() == true) {
            if (sck1.empty() == true) {return -1;}
            // 将sck1中的元素都放到sck2中
            while (sck1.empty() != true) {
                sck2.push(sck1.top());
                sck1.pop();
            }//while
        }//if
        // 返回队列头元素
        int res = sck2.top();
        sck2.pop();
        return res;
    }
};
```
## 树的遍历

### 后序遍历

```cpp
#include <iostream>
#include <vector>
#include <stack>
using namespace std;
class Solution {
public:
    // 后序遍历
    vector<int> postorderTraversal(TreeNode* root) {
        if (root==NULL) return vector<int>{};
        vector<int> ans;
        stack<TreeNode*> st;
        st.push(root);
        while (st.size()){
            root=st.top();
            st.pop();
            if (root==NULL){ //当前节点为空的时候,处理栈顶部的节点
                ans.push_back(st.top()->val);
                st.pop();
            }else{ //当前节点非空的时候压入栈中,待处理
                st.push(root); //*根*
                st.push(NULL); //用来标记未处理的节点,压入栈中
                if (root->right) st.push(root->right); //*右*,有则压入栈中
                if (root->left) st.push(root->left);   //*左*,有则压入栈中
            }//if
        }//while
        return ans;
    }
};
```
### 层序遍历
```cpp
#include <iostream>
#include <queue>
#include <vector>
using namespace std;
class Solution {
public:
    // 层序遍历
    vector<int> levelOrder(TreeNode* root) {
        if(root==NULL) return vector<int>{};
        vector<int> res;
        queue<TreeNode*> qu;
        qu.push(root);
        while(!qu.empty()){
            vector<int> temp;
            int len=qu.size();
            for(int i=0;i<len;i++){
                TreeNode* node=qu.front();
                qu.pop();
                temp.push_back(node->val);
                if(node->left) qu.push(node->left);
                if(node->right) qu.push(node->right);
            }//for
            res.insert(res.end(),temp.begin(),temp.end());
        }//while
        return res;
    }
};
```

## 树的递归应用
### 递归求二叉树深度
```cpp
#include <iostream>
#define max(a,b) (a>b?a:b);
using namespace std;
class Solution {
public:
    //递归,求二叉树深度
    int maxDepth(TreeNode *root) {
        return getDepth(root);
    }
    int getDepth(TreeNode *node) {
        //确定递归函数的参数和返回值：参数就是传入树的根节点,返回就返回这棵树的深度,所以返回值为int类型.
        //确定终止条件：如果为空节点的话,就返回0,表示高度为0.
        if (node==NULL) return 0;
        //确定单层递归的逻辑：
        int leftdepth = getDepth(node->left);       // 左
        int rightdepth = getDepth(node->right);     // 右
        //#define max(a,b) (a>b?a:b);
        int depth = 1 + max(leftdepth, rightdepth); // 中
        return depth;
    }
};
```
### 是否是平衡的二叉树
```cpp
#include <iostream>
#define max(a,b) (a>b?a:b);
using namespace std;
class Solution {
public:
    // 返回以该节点为根节点的二叉树的高度,如果不是二叉搜索树了则返回-1
    int getDepth(TreeNode *node) {
        if (node == NULL) {return 0;}

        int leftDepth = getDepth(node->left);
        if (leftDepth == -1) {return -1;}// 说明左子树已经不是二叉平衡树

        int rightDepth = getDepth(node->right);
        if (rightDepth == -1) {return -1;}// 说明右子树已经不是二叉平衡树
        
        if (abs(leftDepth-rightDepth)>1) {//返回值
            return -1;
        }else{
            return 1 + max(leftDepth,rightDepth);
        }
    }
    //给定一个二叉树,判断它是否是高度平衡的二叉树.
    bool isBalanced(TreeNode *root) {
        return getDepth(root) == -1 ? false : true;
    }
};
```

## 图
### 课程顺序
- 现在总共有 numCourses 门课需要选，记为 0 到 numCourses-1。
- 给定一个数组 prerequisites，它的每一个元素prerequisites[i]表示两门课程之间的先修顺序。
- 例如prerequisites[i]=[ai, bi]表示想要学习课程ai，需要先完成课程bi。
- 请根据给出的总课程数  numCourses 和表示先修顺序的 prerequisites 得出一个可行的修课序列。
- 可能会有多个正确的顺序，只要任意返回一种就可以了。如果不可能完成所有课程，返回一个空数组。
- 示例 1:
    - 输入: numCourses = 2, prerequisites = [[1,0]] 
    - 输出: [0,1]
    - 解释: 总共有 2 门课程。要学习课程 1，你需要先完成课程 0。因此，正确的课程顺序为 [0,1] 。
- 示例 2:
    - 输入: numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
    - 输出: [0,1,2,3] or [0,2,1,3]
    - 解释: 总共有 4 门课程。要学习课程 3，你应该先完成课程 1 和课程 2。并且课程 1 和课程 2 都应该排在课程 0 之后。因此，一个正确的课程顺序是 [0,1,2,3] 。另一个正确的排序是 [0,2,1,3] 。
- 示例 3:
    - 输入: numCourses = 1, prerequisites = [] 
    - 输出: [0]
    - 解释: 总共 1 门课，直接修第一门课就可。
```cpp
class Solution {
private:
    vector<vector<int>> edges;
    vector<int> indeg;
    vector<int> res;
public:
    //经典DAG拓扑排序算法，借助队列实现BFS
    vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites){
        edges.resize(numCourses);
        indeg.resize(numCourses);
        for(const auto& info:prerequisites){
            edges[info[1]].push_back(info[0]);
            ++indeg[info[0]];
        }
        queue<int> q;
        for(int i = 0;i<numCourses;i++){
            if(indeg[i] == 0){
                q.push(i);
            }
        }
        while(!q.empty()){
            int u = q.front();
            q.pop();
            res.push_back(u);
            for(int v:edges[u]){
                --indeg[v];
                if(indeg[v] == 0){
                    q.push(v);
                }
            }
        }
        if(res.size() != numCourses){
            return vector<int>{};
        }
        return res;
    }
};
```

### 矩阵中的路径
- 请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。
- 路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，向下移动一个格子。
- 如果一条路径经过了矩阵中的某一个格子，则之后不能再次进入这个格子。
- 注意：
- 输入的路径不为空；
- 所有出现的字符均为大写英文字母；
- 样例
- matrix=
- [
-   ["A","B","C","E"],
-   ["S","F","C","S"],
-   ["A","D","E","E"]
- ]
- str="BCCE" , return "true" 
- str="ASAE" , return "false"
```cpp
//(DFS) O(n23k)O(n23k)
//在深度优先搜索中，最重要的就是考虑好搜索顺序。
//我们先枚举单词的起点，然后依次枚举单词的每个字母。
//过程中需要将已经使用过的字母改成一个特殊字母，以避免重复使用字符。
//时间复杂度分析：单词起点一共有 n2n2 个，单词的每个字母一共有上下左右四个方向可以选择，但由于不能走回头路，所以除了单词首字母外，仅有三种选择。所以总时间复杂度是 O(n23k)O(n23k)。
class Solution {
public:
    bool hasPath(vector<vector<char>>& matrix, string str) {
        for (int i = 0; i < matrix.size(); i ++ )
            for (int j = 0; j < matrix[i].size(); j ++ )
                if (dfs(matrix, str, 0, i, j))
                    return true;
        return false;
    }

    bool dfs(vector<vector<char>> &matrix, string &str, int u, int x, int y) {
        if (matrix[x][y] != str[u]) return false;
        if (u == str.size() - 1) return true;
        int dx[4] = {-1, 0, 1, 0}, dy[4] = {0, 1, 0, -1};
        char t = matrix[x][y];
        matrix[x][y] = '*';
        for (int i = 0; i < 4; i ++ ) {
            int a = x + dx[i], b = y + dy[i];
            if (a >= 0 && a < matrix.size() && b >= 0 && b < matrix[a].size()) {
                if (dfs(matrix, str, u + 1, a, b)) return true;
            }
        }
        matrix[x][y] = t;
        return false;
    }
};
```

### N皇后问题
- n−皇后问题是指将 n 个皇后放在 n×n 的国际象棋棋盘上，使得皇后不能相互攻击到，即任意两个皇后都不能处于同一行、同一列或同一斜线上。
- 现在给定整数 n，请你输出`所有`的满足条件的棋子摆法。
```cpp
class Solution {
public:
    char q[11][11];//存储棋盘
    bool dg[22], udg[22], cor[11];//点对应的两个斜线以及列上是否有皇后
    vector<vector<string>> res;
    vector<vector<string>> solveNQueens(int n) {
        for (int i = 0; i < n; i ++ )
            for (int j = 0; j < n; j ++ )
                q[i][j] = '.';
        dfs(0,n); 
        return res;
    }

    void dfs(int r,int n){
        if(r == n){//放满了棋盘，输出棋盘
            vector<string> temp1;
            for (int i = 0; i < n; i ++ ){
                string t;
                for (int j = 0; j < n; j ++ ){
                    t+=q[i][j];
                }
                temp1.push_back(t);
            }
            res.push_back(temp1);
            return;
        }
        for(int i = 0; i < n; i++){//第 r 行，第 i 列 是否放皇后
            if(!cor[i] && !dg[i + r] && !udg[n - i + r]){//不冲突，放皇后
                q[r][i] = 'Q';
                cor[i] = dg[i + r] = udg[n - i + r] = 1;//对应的 列， 斜线 状态改变
                dfs(r + 1,n);//处理下一行
                cor[i] = dg[i + r] = udg[n - i + r] = 0;//恢复现场
                q[r][i] = '.';
            }
        }
    }
};
```

### 小镇的法官
- 在一个小镇里,按从1到n为n个人进行编号.传言称,这些人中有一个是小镇上的秘密法官.
- 如果小镇的法官真的存在,那么：
    - 1.小镇的法官不相信任何人.
    - 2.每个人(除了小镇法官外)都信任小镇的法官.
    - 3.只有一个人同时满足条件1和条件2.
- 给定数组trust,该数组由信任对trust[i]=[a,b]组成,表示编号为a的人信任编号为b的人.
- 如果小镇存在秘密法官并且可以确定他的身份,请返回该法官的编号.否则,返回-1.
```cpp
#include <iostream>
#include <unordered_map>
using namespace std;
class Solution {
public:
    //遍历数组,记录每个点的出度和入度.
    //当此点的出度为0,入度为n-1,便是所求答案,无结果返回-1.
    using pii = pair<int, int>;
    int findJudge(int n, vector<vector<int> > &trust) {
        vector<pii> graph(n + 1, pii(0, 0));
        for (auto &v:trust) {
            graph[v[0]].first++;
            graph[v[1]].second++;
        }
        for (int i = 1; i <= n; i++) {
            if (graph[i].first == 0 && graph[i].second == n - 1)return i;
        }
        return -1;
    }
};
```

### HDU 六度分离(flord)
- 1967年,美国著名的社会学家斯坦利・米尔格兰姆提出了一个名为“小世界现象(small world phenomenon)”的著名假说,大意是说,任何2个素不相识的人中间最多只隔着6个人,即只用6个人就可以将他们联系在一起,因此他的理论也被称为 “六度分离”理论(six degrees of separation).虽然米尔格兰姆的理论屡屡应验,一直也有很多社会学家对其兴趣浓厚,但是在30多年的时间里,它从来就没有得到过严谨的证明,只 是一种带有传奇色彩的假说而已.
- Input
    - 本题目包含多组测试,请处理到文件结束.
    - 对于每组测试,第一行包含两个整数N,M(0<N<100,0<M<200),分别代表HDU里的人数(这些人分别编成0~N-1号),以及他们之间的关系.
    - 接下来有M行,每行两个整数A,B(0<=A,B<N)表示HDU里编号为A和编号B的人互相认识.
    - 除了这M组关系,其他任意两人之间均不相识.
- Output
    - 对于每组测试,如果数据符合“六度分离”理论就在一行里输出"Yes",否则输出"No".
```cpp
//佛洛依德算法 + 暴力遍历一遍map数组的数值是否都<=7
//只要有不符合的就break，并且printf（No）; 否则打印 Yes 。
#include <cstdio>
#include <iostream>
using namespace std;
#define ff 999999
int map[105][105];
int n;

void flord() {
    int i, j, k;
    int flag = 1;
    for (k = 0; k < n; k++) {
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                if (map[i][k] != ff && map[k][j] != ff 
                    && map[i][k] + map[k][j] < map[i][j]) {
                    map[i][j] = map[i][k] + map[k][j];
                }
            }
        }
    }
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (map[i][j] > 7) {
                flag = 0;
                break;
            }
        }
        if (flag == 0) break;
    }
    if (flag == 1) cout << "Yes" << endl;
    else cout << "No" << endl;
}

int main() {
    int m;
    int i, j;
    int u, v;
    while (scanf("%d %d", &n, &m) != EOF) {
        for (i = 0; i < n; i++) {//初始化map
            for (j = 0; j < n; j++) {
                if (i == j) {
                    map[i][j] = 0;
                } else {
                    map[i][j] = ff;
                }
            }
        }
        for (i = 0; i < m; i++) {
            scanf("%d %d", &u, &v);
            map[u][v] = 1;
            map[v][u] = 1;
        }
        flord();
    }
    return 0;
}
```

## 哈希表
### 快乐数
- 快乐数定义为：对于一个正整数,每一次将该数替换为它每个位置上的数字的平方和,然后重复这个过程直到这个数变为1,也可能是无限循环但始终变不到1.如果可以变为1,那么这个数就是快乐数.
- 如果n是快乐数就返回True；不是,则返回False.
- 输入：19
- 输出：true
- 解释：
    - 1^2 + 9^2 = 82
    - 8^2 + 2^2 = 68
    - 6^2 + 8^2 = 100
    - 1^2 + 0^2 + 0^2 = 1
```cpp
#include <iostream>
#include <unordered_set>
using namespace std;
class Solution {
public:
    //使用哈希法,来判断这个sum是否重复出现,如果重复了就是return false,否则一直找到sum为1为止.
    //判断sum是否重复出现就可以使用unordered_set.
    // 取数值各个位上的单数之和
    int getSum(int n) {
        int sum = 0;
        while (n) {
            sum += (n % 10) * (n % 10);
            n /= 10;
        }//while
        return sum;
    }
    bool isHappy(int n) {
        unordered_set<int> set;
        while (1) {
            int sum = getSum(n);
            if (sum == 1) {
                return true;
            }
            // 如果这个sum曾经出现过,说明已经陷入了无限循环了,立刻return false
            if (set.find(sum) != set.end()) {
                return false;
            } else {
                set.insert(sum);
            }
            n = sum;
        }
    }
};
```

### 两数之和
- 给定一个整数数组nums和一个目标值target,请你在该数组中找出和为目标值的那两个整数,并返回他们的数组下标.
- 你可以假设每种输入只会对应一个答案.但是,数组中同一个元素不能使用两遍.
- 示例:
- 给定 nums = [2, 7, 11, 15], target = 9
- 因为 nums[0] + nums[1] = 2 + 7 = 9
- 所以返回 [0, 1]
```cpp
#include <iostream>
#include <unordered_map>
using namespace std;
class Solution {
public:
    vector<int> twoSum(vector<int> &nums, int target) {
        unordered_map<int, int> map;
        for (int i = 0; i < nums.size(); i++) {
            auto iter = map.find(target - nums[i]);
            if (iter != map.end()) {
                return {iter->second, i};
            }
            map.insert(pair<int, int>(nums[i], i));
        }
        return {};
    }
};
```

