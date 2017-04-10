#include <stdio.h>

#include "stack.h"

int main(int argc, char *argv[])
{
  Stack s(8);

  int i;
  double d;

  s.push((int) 1);
  s.push((double) 3.4);
  s.push((int) 2);
  s.pop_bottom(i);
  printf("%i\n", i);
  s.pop(i);
  printf("%i\n", i);
  s.push((double) 4.8);
  s.pop_bottom(d);
  printf("%f\n", d);
  s.pop_bottom(d);
  printf("%f\n", d);
  printf("size = %i\n", s.get_size());
}
