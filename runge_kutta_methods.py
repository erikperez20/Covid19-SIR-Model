import numpy as np 


#dt es el time step (h)
# Para ecuaciones diferenciales de primer grado 
def euler_grado_1(t0, tmax, dt, y0, f):
    t = [t0]
    y = [y0]
    for i in range(int(tmax / dt)):
        t.append(t[i] + dt)
        y.append(y[i] + dt * f(y[i],t[i]))
    return np.array(t), np.array(y)

#Para ecuaciones diferenciales de segundo grado 
def euler_grado_2(t0,tmax,dt,y0,yprime0,func):
    t=[t0]
    y=[y0]
    yprime=[yprime0]
    for i in range(int(tmax/dt)):
        t.append(t[i] + dt)
        yprime.append(yprime[i] + func(y[i],t[i]) * dt)
        y.append(y[i]+yprime[i]*dt)
    return np.array(t), np.array(yprime), np.array(y)

#Ecuaciones diferenciales de segundo grado con modificacion de cromer 
def euler_cromer(t0, tmax, dt, y0, yprime0, func):
    t = [t0]
    yprime = [yprime0]
    y = [y0]
    for i in range(int(tmax / dt)):
        t.append(t[i] + dt)
        yprime.append(yprime[i] + func(y[i],t[i]) * dt)
        y.append(y[i] + yprime[i + 1] * dt) #Modificacion de cromer
    return np.array(t), np.array(yprime), np.array(y)

#Ecuacion diferencial de primer grado

def runge_kutta_2th_order_01(t0, tmax, dt, y0, f):
    t = [t0]
    y = [y0]
    for i in range(int(tmax / dt)):
        t.append(t[i] + dt)
        k1 = f(y[i],t[i])
        y1 = y[i] + k1 * dt/2 
        k2 = f(y1, t[i] + dt/2)
        y.append(y[i] + k2 * dt)
    return np.array(t), np.array(y)

#Ecuacion diferencial de segundo grado 
def runge_kutta_2th_order_02(t0,tmax,dt,q10,q20,F,a,b,c):

    t=[t0]
    q1=[q10]   # y(0)
    q2=[q20]   # y'(0)

    A=np.array([[0,-c/a],[1,-b/a]])
    B=np.array([0,1/a])

    for i in range(int(tmax/dt)):
        t.append(t[i] + dt)
        q=np.array([[q1[i]],[q2[i]]])
        k1 = sum(A * q) + B * F(t[i])
 
        qp1 = q + np.array([[k1[0]],[k1[1]]]) * dt/2

        k2 = sum(A * qp1) + B * F(t[i] + dt/2)

        P= q + np.array([[k2[0]],[k2[1]]]) * dt
        P_aux= sum(P * [[1,0],[0,1]])
        q1.append(P_aux[0])
        q2.append(P_aux[1])

    return np.array(t), np.array(q2), np.array(q1)

# Runge kutta 3th order for 1st order differential equation 

def runge_kutta_3th_order_01(t0, tmax, dt, y0, f):
    t = [t0]
    y = [y0]
    for i in range(int(tmax / dt)):
        t.append(t[i] + dt)
        k1 = f(y[i],t[i])
        y1 = y[i] + k1 * dt/2 
        k2 = f(y1, t[i] + dt/2)
        y2 = y[i] + k2 * dt/2
        k3 = f(y2, t[i] + dt/2)
        y.append(y[i] + k3 * dt)
    return np.array(t), np.array(y)

# Runge kutta 3th order for 2nd order differential equation

def runge_kutta_3th_order_02(t0,tmax,dt,q10,q20,F,a,b,c):

    t=[t0]
    q1=[q10]   # y(0)
    q2=[q20]   # y'(0)

    A=np.array([[0,-c/a],[1,-b/a]])
    B=np.array([0,1/a])

    for i in range(int(tmax/dt)):
        t.append(t[i] + dt)
        q=np.array([[q1[i]],[q2[i]]])
        k1 = sum(A * q) + B * F(t[i])
 
        qp1 = q + np.array([[k1[0]],[k1[1]]]) * dt/2

        k2 = sum(A * qp1) + B * F(t[i] + dt/2)

        qp2 = q + np.array([[k2[0]],[k2[1]]]) * dt/2

        k3 = sum(A * qp2) + B * F(t[i] + dt/2)

        P= q + np.array([[k3[0]],[k3[1]]]) * dt
        P_aux= sum(P * [[1,0],[0,1]])
        q1.append(P_aux[0])
        q2.append(P_aux[1])

    return np.array(t), np.array(q2), np.array(q1)

# Runge kutta 4th order for 1st order differential equation 

def runge_kutta_4th_order_01(t0, tmax, dt, y0, f):
    t = [t0]
    y = [y0]
    for i in range(int(tmax / dt)):
        t.append(t[i] + dt)
        k1 = f(y[i],t[i])
        y1 = y[i] + k1 * dt/2 
        k2 = f(y1, t[i] + dt/2)
        y2 = y[i] + k2 * dt/2
        k3 = f(y2, t[i] + dt/2)
        y3 = y[i] + k3 * dt/2
        k4 = f(y3, t[i] + dt/2)
        y.append(y[i] + (1/6)*dt*(k1+2*k2+2*k3+k4))
    return np.array(t), np.array(y)

# Runge kutta 4th order for 2nd order differential equation

def runge_kutta_4th_order_02(t0,tmax,dt,q10,q20,F,a,b,c):

    t=[t0]
    q1=[q10]   # y(0)
    q2=[q20]   # y'(0)

    A=np.array([[0,-c/a],[1,-b/a]])
    B=np.array([0,1/a])

    for i in range(int(tmax/dt)):
        t.append(t[i] + dt)
        q=np.array([[q1[i]],[q2[i]]])

        k1 = sum(A * q) + B * F(t[i])
        qp1 = q + np.array([[k1[0]],[k1[1]]]) * dt/2
        k2 = sum(A * qp1) + B * F(t[i] + dt/2)
        qp2 = q + np.array([[k2[0]],[k2[1]]]) * dt/2
        k3 = sum(A * qp2) + B * F(t[i] + dt/2)
        qp3 = q + np.array([[k3[0]],[k3[1]]]) * dt/2
        k4 = sum(A * qp3) + B * F(t[i] + dt/2)

        P= q + np.array([[k4[0]],[k4[1]]]) * dt
        P_aux= sum(P * [[1,0],[0,1]])
        q1.append(P_aux[0])
        q2.append(P_aux[1])

    return np.array(t), np.array(q2), np.array(q1)

# Runge kutta 4th order for 3 coupled first order differential equations 

# y1'(t) = f1(t,y1,y2,y3) 
# y2'(t) = f2(t,y1,y2,y3)
# y3'(t) = f3(t,y1,y2,y3)

def runge_kutta_4th_order_3_coupled(t0,tmax,dt,y0,f1,f2,f3):
    
    # 3 initial conditions

    t=[t0]
    y1 = [y0[0]]
    y2 = [y0[1]]
    y3 = [y0[2]]

    for i in range(int(tmax/dt)):

        t.append(t[i] + dt)
        
        # k1 vector with 3 components for each y equation
        k11 = f1(t[i],y1[i],y2[i],y3[i])
        k12 = f2(t[i],y1[i],y2[i],y3[i])
        k13 = f3(t[i],y1[i],y2[i],y3[i])

        # k2 vector with 3 components for each y equation
        k21 = f1(t[i] + 0.5*dt , y1[i]+0.5*dt*k11 , y2[i]+0.5*dt*k12 , y3[i]+0.5*dt*k13)
        k22 = f2(t[i] + 0.5*dt , y1[i]+0.5*dt*k11 , y2[i]+0.5*dt*k12 , y3[i]+0.5*dt*k13)
        k23 = f3(t[i] + 0.5*dt , y1[i]+0.5*dt*k11 , y2[i]+0.5*dt*k12 , y3[i]+0.5*dt*k13)

        # k3 vector with 3 components for each y equation
        k31 = f1(t[i] + 0.5*dt , y1[i]+0.5*dt*k21 , y2[i]+0.5*dt*k22 , y3[i]+0.5*dt*k23)
        k32 = f2(t[i] + 0.5*dt , y1[i]+0.5*dt*k21 , y2[i]+0.5*dt*k22 , y3[i]+0.5*dt*k23)
        k33 = f3(t[i] + 0.5*dt , y1[i]+0.5*dt*k21 , y2[i]+0.5*dt*k22 , y3[i]+0.5*dt*k23)
    
        # k4 vector with 3 components for each y equation
        k41 = f1(t[i] + dt , y1[i] + dt*k31 , y2[i] + dt*k32 , y3[i] + dt*k33)
        k42 = f2(t[i] + dt , y1[i] + dt*k31 , y2[i] + dt*k32 , y3[i] + dt*k33)
        k43 = f3(t[i] + dt , y1[i] + dt*k31 , y2[i] + dt*k32 , y3[i] + dt*k33)

        y1.append(y1[i] + (1/6) * dt * (k11 + 2*k21 + 2*k31 + k41))
        y2.append(y2[i] + (1/6) * dt * (k12 + 2*k22 + 2*k32 + k42))
        y3.append(y3[i] + (1/6) * dt * (k13 + 2*k23 + 2*k33 + k43))

    return np.array(t),np.array(y1),np.array(y2),np.array(y3)













# Runge kutta 4th order for n coupled first order differential equations 

# y1'(t) = f1(t,y1,y2,y3,...,yn) 
# y2'(t) = f2(t,y1,y2,y3,...,yn)
# y3'(t) = f3(t,y1,y2,y3,...,yn)
#        .
#        .
#        .
# yn'(t) = fn(t,y1,y2,y3,...,yn)