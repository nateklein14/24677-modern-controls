clear
clc

syms x_dot y_dot phi_dot X Y phi C delta f F l_r l_f I_z m g y x

s1 = [y ;
      y_dot ;
      phi ;
      phi_dot];
s1_dot = [y_dot ;
          -phi_dot*x_dot + (2*C) / m * cos(delta) * (delta - (y_dot + l_f*phi_dot)/x_dot) - (y_dot - l_r*phi_dot)/x_dot ;
          phi_dot
          (2*l_f*C)/I_z * (delta - (y_dot + l_f*phi_dot)/x_dot) - (2*l_r*C)/I_z*(-(y_dot - l_r*phi_dot)/x_dot)];

s2 = [x ;
      x_dot];
s2_dot = [x_dot ;
          phi_dot*y_dot + (1/m) * (F - f*m*g)];

u = [delta ;
     F];

A1 = simplify(jacobian(s1_dot, s1));
B1 = simplify(jacobian(s1_dot, u));

e1 = solve(s1_dot == [0;0;0;0], [s1 ; u]);

A1 = subs(A1, [s1 ; u], [e1.y ; e1.y_dot ; e1.phi ; e1.phi_dot ; e1.delta ; e1.F]);
B1 = subs(B1, [s1 ; u], [e1.y ; e1.y_dot ; e1.phi ; e1.phi_dot ; e1.delta ; e1.F]);

A2 = simplify(jacobian(s2_dot, s2));
B2 = simplify(jacobian(s2_dot, u));

e2 = solve(s2_dot == [0;0], [s2 ; u]);

A2 = subs(A2, [s2 ; u], [e2.x ; e2.x_dot ; e2.delta ; e2.F]);
B2 = subs(B2, [s2 ; u], [e2.x ; e2.x_dot ; e2.delta ; e2.F]);

disp(A1)
disp(B1)
disp(A2)
disp(B2)