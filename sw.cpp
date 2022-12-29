#include <bits/stdc++.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#define MAXSEQ 1000
#define GAP_CHAR '-'
#define SCORE_FOR_MATCH 1
#define PUNISH_FOR_MISMATCH -1e8
#define PUNISH_FOR_SKIP 0

using namespace std;
// 对空位的罚分是线性的
struct Unit {
  int W1;   // 是否往上回溯一格
  int W2;   // 是否往左上回溯一格
  int W3;   // 是否往左回溯一格
  float M;  // 得分矩阵第(i, j)这个单元的分值，即序列s(1,...,i)与序列r(1,...,j)比对的最高得分
};
typedef struct Unit *pUnit;

void strUpper(char *s);
float max4(float a, float b, float c, float d);
float getFScore(char a, char b);
void printAlign(pUnit **a, const int i, const int j, char *s, char *r, char *saln, char *raln, int n);
void align(char *s, char *r);
set<string> resultSet[MAXSEQ][MAXSEQ];
int main(int argc, char *argv[]) {
  char input_path[100];
  strcpy(input_path, "/home/chijj/gpu/example/double/1.txt");
  for (int i = 0; i < argc; i++) {
    if (!strcmp(argv[i], "--input") || !strcmp(argv[i], "-i")) {
      strcpy(input_path, argv[++i]);
    }
  }
    freopen(input_path, "r", stdin);
  char s[MAXSEQ];
  char r[MAXSEQ];
  int n;
  scanf("%d", &n);
  // printf("The 1st seq: ");
  scanf("%s", s);
  // printf("The 2nd seq: ");
  scanf("%s", r);
  auto _outt =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
  std::cout << "[PF] bg fast_sw " << std::fixed << std::setprecision(6) << (double)(_outt.count() * 0.000001) << "\n";
  // std::shared_ptr<HeteroSubgraph> subg(new HeteroSubgraph);
  align(s, r);
  _outt = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
  std::cout << "[PF] ed fast_sw " << std::fixed << std::setprecision(6) << (double)(_outt.count() * 0.000001) << "\n";
  return 0;
}

void strUpper(char *s) {
  while (*s != '\0') {
    if (*s >= 'a' && *s <= 'z') {
      *s -= 32;
    }
    s++;
  }
}

float max4(float a, float b, float c, float d) {
  float f = a > b ? a : b;
  float g = c > d ? c : d;
  return f > g ? f : g;
}

// 替换矩阵：match分值为1，mismatch分值为-inf
// 数组下标是两个字符的ascii码减去65之后的和
float FMatrix[] = {SCORE_FOR_MATCH,
                   0,
                   PUNISH_FOR_MISMATCH,
                   0,
                   SCORE_FOR_MATCH,
                   0,
                   PUNISH_FOR_MISMATCH,
                   0,
                   PUNISH_FOR_MISMATCH,
                   0,
                   0,
                   0,
                   SCORE_FOR_MATCH,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   PUNISH_FOR_MISMATCH,
                   0,
                   PUNISH_FOR_MISMATCH,
                   0,
                   0,
                   0,
                   PUNISH_FOR_MISMATCH,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   SCORE_FOR_MATCH};

float getFScore(char a, char b) { return FMatrix[a + b - 'A' - 'A']; }

// void printAlign(pUnit **a, const int i, const int j, char *s, char *r, char *saln, char *raln, int n)
void printAlign(pUnit **a, const int i, const int j, char *s, char *r, char *saln, char *raln, int n) {
  int k;
  pUnit p = a[i][j];
  if (resultSet[i][j].size() != 0) return;
  if (p->M == 0) {
    // for (k = n - 1; k >= 0; k--)
    //     printf("%c", saln[k]);
    // printf("\n");
    // for (k = n - 1; k >= 0; k--)
    //     printf("%c", raln[k]);
    // printf("\n\n");
    // return;
    // string s = raln;
    resultSet[i][j].insert("");
    return;
  }
  if (p->W1) {  // 向上回溯一格
    // saln[n] = s[i - 1];
    // raln[n] = GAP_CHAR;
    // printAlign(a, i - 1, j, s, r, saln, raln, n + 1);
    printAlign(a, i - 1, j, s, r, saln, raln, n);
    resultSet[i][j].insert(resultSet[i - 1][j].begin(), resultSet[i - 1][j].end());
  }
  if (p->W2) {  // 向左上回溯一格
    saln[n] = s[i - 1];
    raln[n] = r[j - 1];
    printAlign(a, i - 1, j - 1, s, r, saln, raln, n + 1);
    for (auto x : resultSet[i - 1][j - 1]) resultSet[i][j].insert(x + s[i - 1]);
  }
  if (p->W3) {  // 向左回溯一格
    // saln[n] = GAP_CHAR;
    // raln[n] = r[j - 1];
    // printAlign(a, i, j - 1, s, r, saln, raln, n + 1);
    printAlign(a, i, j - 1, s, r, saln, raln, n);
    resultSet[i][j].insert(resultSet[i][j - 1].begin(), resultSet[i][j - 1].end());
  }
}

void align(char *s, char *r) {
  int i, j;
  int m = strlen(s);
  int n = strlen(r);
  float gap = PUNISH_FOR_SKIP;  // 对空位的罚分
  float m1, m2, m3, maxm;
  int maxMatrix;  // 得分矩阵中的最高分
  pUnit **aUnit;
  char *salign;
  char *ralign;
  int cnt = 0;
  // 初始化
  if ((aUnit = (pUnit **)malloc(sizeof(pUnit *) * (m + 1))) == NULL) {
    fputs("Error: Out of space!\n", stderr);
    exit(1);
  }
  for (i = 0; i <= m; i++) {
    if ((aUnit[i] = (pUnit *)malloc(sizeof(pUnit) * (n + 1))) == NULL) {
      fputs("Error: Out of space!\n", stderr);
      exit(1);
    }
    for (j = 0; j <= n; j++) {
      if ((aUnit[i][j] = (pUnit)malloc(sizeof(struct Unit))) == NULL) {
        fputs("Error: Out of space!\n", stderr);
        exit(1);
      }
      aUnit[i][j]->W1 = 0;
      aUnit[i][j]->W2 = 0;
      aUnit[i][j]->W3 = 0;
    }
  }
  aUnit[0][0]->M = 0;
  for (i = 1; i <= m; i++) {
    aUnit[i][0]->M = 0;
  }
  for (j = 1; j <= n; j++) {
    aUnit[0][j]->M = 0;
  }
  // 将字符串都变成大写
  strUpper(s);
  strUpper(r);
  // 动态规划算法计算得分矩阵每个单元的分值
  for (i = 1; i <= m; i++) {
    for (j = 1; j <= n; j++) {
      m1 = aUnit[i - 1][j]->M + gap;
      m2 = aUnit[i - 1][j - 1]->M + getFScore(s[i - 1], r[j - 1]);
      m3 = aUnit[i][j - 1]->M + gap;
      maxm = max4(m1, m2, m3, 0);
      aUnit[i][j]->M = maxm;
      if (maxm != 0) {
        if (m1 == maxm) aUnit[i][j]->W1 = 1;
        if (m2 == maxm) aUnit[i][j]->W2 = 1;
        if (m3 == maxm) aUnit[i][j]->W3 = 1;
      }
    }
  }
  /*
      // 打印得分矩阵
      for (i = 0; i <= m; i++) {
          for (j = 0; j <= n; j++)
              printf("%f ", aUnit[i][j]->M);
          printf("\n");
      }
  */
  // 求取得分矩阵中的最高分
  maxMatrix = 0;
  for (i = 1; i <= m; i++) {
    for (j = 1; j <= n; j++) {
      if (aUnit[i][j]->M > maxMatrix) maxMatrix = aUnit[i][j]->M;
    }
  }
  // printf("%d\n", maxMatrix);
  // 打印最优比对结果，如果有多个，全部打印
  // 递归法
  if (maxMatrix == 0) {
    fputs("No seq aligned.\n", stdout);
  } else {
    if ((salign = (char *)malloc(sizeof(char) * (m + n + 1))) == NULL) {
      fputs("Error: Out of space!\n", stderr);
      exit(1);
    }
    if ((ralign = (char *)malloc(sizeof(char) * (m + n + 1))) == NULL) {
      fputs("Error: Out of space!\n", stderr);
      exit(1);
    }
    for (i = m; i >= 1; i--)
      for (j = n; j >= 1; j--)
        if (aUnit[i][j]->M == maxMatrix) {
          printAlign(aUnit, i, j, s, r, salign, ralign, 0);
          cnt += resultSet[i][j].size();
        }

    // 释放内存
    free(salign);
    free(ralign);
  }
  // cout << cnt << endl;
  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
      free(aUnit[i][j]);
    }
    free(aUnit[i]);
  }
  free(aUnit);
}