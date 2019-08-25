#include <stdio.h>
#include <cmath>

#include "stack.h"

int main(int argc, char *argv[])
{
  Stack s(8);

  int i, j;
  double d;

  printf("push(1)\n");
  s.push((int) 1);
  printf("size = %lu, %lu\n", s.get_size(), s.get_buffer_size());
#ifndef DEBUG_STACK
  assert(s.get_size() == sizeof(int));
#endif
  printf("push(3.4)\n");
  s.push((double) 3.4);
  printf("size = %lu, %lu\n", s.get_size(), s.get_buffer_size());
#ifndef DEBUG_STACK
  assert(s.get_size() == sizeof(int) + sizeof(double));
#endif
  printf("push(2)\n");
  s.push((int) 2);
  printf("size = %lu, %lu\n", s.get_size(), s.get_buffer_size());
#ifndef DEBUG_STACK
  assert(s.get_size() == 2*sizeof(int) + sizeof(double));
#endif
  s.pop_bottom(i);
  printf("pop_bottom = %i\n", i);
  assert(i == 1);
  s.pop(i);
  printf("pop = %i\n", i);
  assert(i == 2);
  printf("push(4.8)\n");
  s.push((double) 4.8);
  printf("size = %lu, %lu\n", s.get_size(), s.get_buffer_size());
#ifndef DEBUG_STACK
  assert(s.get_size() == 2*sizeof(double));
#endif
  s.pop_bottom(d);
  printf("pop_bottom = %f\n", d);
  assert(std::abs(d - 3.4) < 1e-6);
  s.pop_bottom(d);
  printf("pop_bottom = %f\n", d);
  assert(std::abs(d - 4.8) < 1e-6);
  printf("size = %lu, %lu\n", s.get_size(), s.get_buffer_size());
  printf("push(2, 3)\n");
  s.push((int) 2, (int) 3);
  s.pop_bottom(i, j);
  printf("pop_bottom = %i, %i\n", i, j);
  assert(i == 2);
  assert(j == 3);
}
