lr = 1.39;
lf = 1.55;
Ca = 20000;
Iz = 25854;
m = 1888.6;
g = 9.81;

x_dot = [2 5 8];
disp(["x_dot" "controllable?" "observable?"])

for i=1:length(x_dot)
    controllable = true;
    observable = true;
    A = [0 1 0 0 ;
         0 -4*Ca/(m*x_dot(i)) 4*Ca/m -2*Ca*(lf-lr)/(m*x_dot(i)) ;
         0 0 0 1 ;
         0 -2*Ca*(lf-lr)/(Iz*x_dot(i)) 2*Ca*(lf-lr)/Iz -2*Ca*(lf^2+lr^2)/(Iz*x_dot(i))];
    B = [0 0 ;
         2*Ca/m 0 ;
         0 0 ;
         2*Ca*lf/Iz 0];
    e = eig(A);
    for j=1:length(e)
        if rank([e(j)*eye(4) - A B]) < 4
            controllable = false;
        end
        if rank([e(j)*eye(4) - A ; [1 1 1 1]]) < 4
            observable = false;
        end
    end
    disp([x_dot(i) controllable observable])
end