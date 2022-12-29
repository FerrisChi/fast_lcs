#include <chrono>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
const int MAX_LENGTH = 10000;
char x[MAX_LENGTH + 2], y[MAX_LENGTH + 2];
int f[MAX_LENGTH + 2][MAX_LENGTH + 2];
int max(int x, int y) { return (x > y) ? x : y; }
int main(int argc, char *argv[]) {
  char input_path[100];
  strcpy(input_path, "/home/chijj/gpu/example/double/1.txt");
  for (int i = 0; i < argc; i++) {
    if (!strcmp(argv[i], "--input") || !strcmp(argv[i], "-i")) {
      strcpy(input_path, argv[++i]);
    }
  }
  freopen(input_path, "r", stdin);
  auto _outt =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
  std::cout << "[PF] bg dp " << std::fixed << std::setprecision(6) << (double)(_outt.count() * 0.000001) << "\n";
  int n, m, s;
  // scanf("%d%d\n", &n, &m);
  // for (int i = 1; i <= n; i++) {
  //   x[i] = getchar();
  // }
  // scanf("\n");
  // for (int i = 1; i <= m; i++) {
  //   y[i] = getchar();
  // }
  scanf("%d", &s);
  getchar();
  scanf("%s", x + 1);
  scanf("%s", y + 1);
  n = strlen(x + 1);
  m = strlen(y + 1);
  for (int i = 1; i <= n; i++) {
    for (int j = 1; j <= m; j++) {
      f[i][j] = max(f[i - 1][j - 1] + (x[i] == y[j]), max(f[i][j - 1], f[i - 1][j]));
    }
  }
  printf("LCS is %d\n", f[n][n]);
  _outt = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
  std::cout << "[PF] ed dp " << std::fixed << std::setprecision(6) << (double)(_outt.count() * 0.000001) << "\n";
}