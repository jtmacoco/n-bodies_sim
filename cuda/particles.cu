#include "particles.h"
__device__ float3 gravitationalForce(Particle p1, Particle p2)
{
    float eps = 2.0f; // needed so force doesn't get to big shooting particles everywhere
    float3 p1_pos = p1.getPosition();
    float p1_mass = p1.getMass();

    float3 p2_pos = p2.getPosition();
    float p2_mass = p2.getMass();

    float3 distance = make_float3(p2_pos.x - p1_pos.x, p2_pos.y - p1_pos.y, p2_pos.z - p1_pos.z); // distance between 2 particles

    float r = sqrtf(distance.x * distance.x + distance.y * distance.y + distance.z * distance.z); //  length of the vector

    float3 r_unit = make_float3(distance.x / r, distance.y / r, distance.z / r);
    float force = G * ((p1_mass * p2_mass) / ((r * r) + (eps * eps))); // gravitational force equation

    float3 gF = make_float3(force * r_unit.x, force * r_unit.y, force * r_unit.z);
    return gF;
}
__global__ void sumForces(
    Particle *particles,
    float3 *forces)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= MX_PARTICLES){return;}
    Particle &cur_particle = particles[i];
    float3 force_sum = make_float3(0.0f, 0.0f, 0.0f);
    for (int j = 0; j < MX_PARTICLES; j++)
    {
        if (j == i)
        {
            continue;
        }
        float3 gF = gravitationalForce(cur_particle, particles[j]);
        force_sum.x += gF.x;
        force_sum.y += gF.y;
        force_sum.z += gF.z;
        // printf("x: %f, y: %f \n",force_sum.x,force_sum.y);
    }
    forces[i] = force_sum;
    // cur_particle.applyForce(force_sum, dt);
}
__global__ void applyForces(
    Particle *particles,
    float3 *forces,
    float dt)
{
     int i = blockIdx.x * blockDim.x + threadIdx.x;
     if(i >= MX_PARTICLES){return;}
        particles[i].applyForce(forces[i], dt);
}
void Particles::prepRender()
{

    initSystem();
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Particle) * 2 * MX_PARTICLES, NULL, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
}

/*
void Particles::addParticle(float mass)
{
    float3 pos = randCircle();
    float3 vel = randCircle();
    std::shared_ptr<Particle> p(new Particle(pos,vel,mass));//change this to float3 for pos and vel
    particles.push_back(p);
}
    */
__host__ void Particles::render(float dt)
{
    dt = 0.001; // maybe adjust
    glBindVertexArray(VAO);
    std::vector<float> vertex_data;
    int threadsPerBlock = 1024; 
    //int maxBlocks = 256;
    //int blocksPerGrid =maxBlocks;
    int blocksPerGrid = (MX_PARTICLES + threadsPerBlock - 1) / threadsPerBlock;//cal number of blocks to cover all particles
    sumForces<<<blocksPerGrid,threadsPerBlock>>>(particles,forces);
    applyForces<<<blocksPerGrid, threadsPerBlock>>>(particles, forces, dt);
    cudaDeviceSynchronize();
    for(int i=0; i < MX_PARTICLES; i++){
        float x = particles[i].getPosition().x;
        float y = particles[i].getPosition().y;
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
    glDrawArrays(GL_POINTS, 0, MX_PARTICLES);
    glBindVertexArray(0);
}
void Particles::initParticles()
{
    for (int i = 0; i < MX_PARTICLES; i++)
    {
        float3 vel = randCircle();
        float3 pos = randCircle();
        particles[i] = Particle(pos, vel, 1.0f);
    }
}
/*Initlizing random position and velocities to be within a circle that is of random radius*/
float3 Particles::randCircle()
{
    static std::uniform_real_distribution<float> dist_theta(1.0f, 2.0f * M_PI);
    static std::uniform_real_distribution<float> dist_radius(0.0f, 0.8f);

    float theta = dist_theta(gen);
    float r = dist_radius(gen);
    float3 res = make_float3(std::cos(theta), std::sin(theta), 0.0f);
    res.x *= r;
    res.y *= r;
    res.z *= r;
    return res;
}
/*summing up the forces*/
/*grab average velocity over all particles in the system*/
void Particles::initVel()
{
    avg_vel = make_float3(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < MX_PARTICLES; i++)
    {
        float3 cur_vel = particles[i].getVelocity();
        float cur_mass = particles[i].getMass();

        avg_vel.x += (cur_vel.x * cur_mass);
        avg_vel.y += (cur_vel.y * cur_mass);
        avg_vel.z += (cur_vel.z * cur_mass);
    }
    avg_vel.x /= MX_PARTICLES;
    avg_vel.y /= MX_PARTICLES;
    avg_vel.z /= MX_PARTICLES;
}
/*grab center of mass of the system basically*/
void Particles::initPos()
{
    center_mass = make_float3(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < MX_PARTICLES; i++)
    {
        float3 cur_pos = particles[i].getPosition();
        float cur_mass = particles[i].getMass();
        center_mass.x += (cur_pos.x * cur_mass);
        center_mass.y += (cur_pos.y * cur_mass);
        center_mass.z += (cur_pos.z * cur_mass);
    }
    center_mass.x /= MX_PARTICLES;
    center_mass.y /= MX_PARTICLES;
    center_mass.z /= MX_PARTICLES;
}
/*This is meant to intilize the conditions at the start of the simulation.*/
void Particles::initSystem()
{
    initParticles();
    initVel();
    initPos();
    for (int i = 0; i < MX_PARTICLES; i++)
    {
        particles[i].setPosition((center_mass)); // note not realy setting but doing p pos-= center of mass
        particles[i].setVelocity(avg_vel);       // note not really setting but doing  p vel -= avg_vel
    }
}