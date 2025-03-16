#include "particles.h"
void Particles::prepRender()
{
    initSystem();
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 2 * particles.size(), NULL, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
}
void Particles::addParticle(float mass)
{
    float3 pos = randCircle();
    float3 vel = randCircle();
    std::shared_ptr<Particle> p(new Particle(pos,vel,mass));//change this to float3 for pos and vel
    particles.push_back(p);
}
void Particles::render(float dt)
{
    dt =  0.001;//maybe adjust
    glBindVertexArray(VAO);
    std::vector<float> vertex_data;
    sumForces(dt);
    for (auto &p : particles)
    {
        float x = p->getPosition().x;
        float y = p->getPosition().y;
        vertex_data.push_back(x);
        vertex_data.push_back(y);
    }
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    /*
        note allocating space for the memory in prep can now can just fill the space with
        vertex data. SubData "reserve a specific amount of memory" CH.28
    */
    glBufferSubData(GL_ARRAY_BUFFER, 0, vertex_data.size() * sizeof(GLfloat), vertex_data.data());
    glPointSize(particle_size);
    glDrawArrays(GL_POINTS, 0, particles.size());
    glBindVertexArray(0);
}
/*Initlizing random position and velocities to be within a circle that is of random radius*/
float3 Particles::randCircle(){
    static std::uniform_real_distribution<float> dist_theta(0.0f, 2.0f * M_PI);
    static std::uniform_real_distribution<float> dist_radius(0.0f, 0.8f);

    float theta = dist_theta(gen);  
    float r = dist_radius(gen);       
    float3 res = make_float3(std::cos(theta),std::sin(theta),0.0f);
    res.x*=r;
    res.y*=r;
    res.z*=r;
    return res;
}
/*summing up the forces*/
void Particles::sumForces(float dt)
{
    for (int i = 0; i < int(particles.size()); i++)
    {
        std::shared_ptr<Particle> cur_particle = particles[i];
        float3 force_sum = make_float3(0.0f,0.0f,0.0f);
        for (int j = i+1; j < int(particles.size()); j++)
        {
            if (j == i)
            {
                continue;
            }
            float3 gF = gravitationalForce(*cur_particle, *particles[j]);
            force_sum.x+=gF.x;
            force_sum.y+=gF.y;
            force_sum.z+=gF.z;
            //printf("x: %f, y: %f \n",force_sum.x,force_sum.y);
        }
        cur_particle->applyForce(force_sum, dt);
    }
} 
float3 Particles::gravitationalForce(Particle p1, Particle p2)
{
    float eps = 2.0f;//needed so force doesn't get to big shooting particles everywhere
    float3 p1_pos = p1.getPosition();
    float p1_mass = p1.getMass();

    float3 p2_pos = p2.getPosition();
    float p2_mass = p2.getMass();

    float3 distance = make_float3(p2_pos.x - p1_pos.x, p2_pos.y - p1_pos.y, p2_pos.z - p1_pos.z);//distance between 2 particles

    float r = sqrtf(distance.x * distance.x + distance.y * distance.y + distance.z * distance.z); //  length of the vector

    float3 r_unit = make_float3(distance.x / r, distance.y / r, distance.z / r);
    float force = G * ((p1_mass * p2_mass) /((r * r)+(eps*eps)));//gravitational force equation

    float3 gF = make_float3(force*r_unit.x, force*r_unit.y, force*r_unit.z);
    return gF;
}
/*grab average velocity over all particles in the system*/
void Particles::initVel(){
    avg_vel= make_float3(0.0f,0.0f,0.0f);
    for(auto &p:particles){
        float3 cur_vel = p->getVelocity();
        float cur_mass = p->getMass();

        avg_vel.x += (cur_vel.x *cur_mass);
        avg_vel.y += (cur_vel.y *cur_mass);
        avg_vel.z += (cur_vel.z *cur_mass);
    }
    avg_vel.x /=particles.size();
    avg_vel.y /=particles.size();
    avg_vel.z /=particles.size();
}
/*grab center of mass of the system basically*/
void Particles::initPos(){
    center_mass = make_float3(0.0f,0.0f,0.0f);
    for(auto &p:particles){
        float3 cur_pos = p->getPosition();
        float cur_mass = p->getMass();
        center_mass.x += (cur_pos.x*cur_mass);
        center_mass.y += (cur_pos.y*cur_mass);
        center_mass.z += (cur_pos.z*cur_mass);
    }
    center_mass.x /=particles.size();
    center_mass.y /=particles.size();
    center_mass.z /=particles.size();
}
/*This is meant to intilize the conditions at the start of the simulation.*/
void Particles::initSystem(){
    initVel();
    initPos();
    for(auto &p:particles){
        p->setPosition((center_mass));//note not realy setting but doing p pos-= center of mass
        p->setVelocity(avg_vel);//note not really setting but doing  p vel -= avg_vel
    }
}
