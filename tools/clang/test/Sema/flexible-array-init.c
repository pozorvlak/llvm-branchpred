// RUN: %clang_cc1 -fsyntax-only -pedantic -verify %s
struct one {
  int a;
  int values[]; // expected-note 3{{initialized flexible array member 'values' is here}}
} x = {5, {1, 2, 3}}; // expected-warning{{flexible array initialization is a GNU extension}}

struct one x2 = { 5, 1, 2, 3 }; // expected-warning{{flexible array initialization is a GNU extension}}

void test() {
  struct one x3 = {5, {1, 2, 3}}; // expected-warning{{flexible array initialization is a GNU extension}}
}

struct foo { 
  int x; 
  int y[]; // expected-note 6 {{initialized flexible array member 'y' is here}}
}; 
struct bar { struct foo z; }; // expected-warning {{'z' may not be nested in a struct due to flexible array member}}
     
struct foo a = { 1, { 2, 3, 4 } };        // expected-warning{{flexible array initialization is a GNU extension}}
struct bar b = { { 1, { 2, 3, 4 } } };    // expected-error{{non-empty initialization of flexible array member inside subobject}}
struct bar c = { { 1, { } } };            // // expected-warning{{flexible array initialization is a GNU extension}} \
              // expected-warning{{use of GNU empty initializer extension}} \
              // expected-warning{{zero size arrays are an extension}}
struct foo d[1] = { { 1, { 2, 3, 4 } } };  // expected-warning{{'struct foo' may not be used as an array element due to flexible array member}} \
              // expected-error{{non-empty initialization of flexible array member inside subobject}}

struct foo desig_foo = { .y = {2, 3, 4} };
struct bar desig_bar = { .z.y = { } }; // expected-warning{{use of GNU empty initializer extension}} \
  // expected-warning{{zero size arrays are an extension}}
struct bar desig_bar2 = { .z.y = { 2, 3, 4} }; // expected-error{{non-empty initialization of flexible array member inside subobject}}
struct foo design_foo2 = { .y = 2 }; // expected-error{{flexible array requires brace-enclosed initializer}}

struct point {
  int x, y;
};

struct polygon {
  int numpoints;
  struct point points[]; // expected-note{{initialized flexible array member 'points' is here}}
};
struct polygon poly = { 
  .points[2] = { 1, 2} }; // expected-error{{designator into flexible array member subobject}}

// PR3540
struct X {
  int a;
  int b;
  char data[];
};

struct Y {
  int a:4;
  int b:4;
  int c;
  int d;
  int e;
  struct X xs[]; // expected-warning{{'struct X' may not be used as an array element due to flexible array member}}
};
