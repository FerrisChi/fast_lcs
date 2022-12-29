#include <stdlib.h>

#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
using namespace std;
int main(int argc, char *argv[]) {
  char output_path[100];
  strcpy(output_path, "/home/chijj/gpu/test/seq.txt");
  int n_seq = 2, seq_len = 100;
  for (int i = 0; i < argc; i++) {
    if (!strcmp(argv[i], "--output") || !strcmp(argv[i], "-o")) {
      strcpy(output_path, argv[++i]);
    } else if (!strcmp(argv[i], "--num_seq") || !strcmp(argv[i], "-n")) {
      string s(argv[++i]);
      n_seq = stoi(s);
    } else if (!strcmp(argv[i], "--seq_len") || !strcmp(argv[i], "-l")) {
      string s(argv[++i]);
      seq_len = stoi(s);
    }
  }
  freopen(output_path, "w", stdout);
  srand(time(0));
  printf("%d\n", n_seq);
  for (int o = 1; o <= n_seq; o++) {
    for (int i = 1; i <= seq_len; i++) {
      int ch = rand() % 4;
      switch (ch) {
        case 0:
          putchar('A');
          break;
        case 1:
          putchar('T');
          break;
        case 2:
          putchar('C');
          break;
        case 3:
          putchar('G');
          break;

        default:
          break;
      }
    }
    printf("\n");
  }
}