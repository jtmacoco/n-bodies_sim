#pragma once
#include <vector>
#include <random>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include "particle.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <iostream>
// #define G 6.674e-11
#define G 1.0f // set to 1 because actual G constant is super small
#define MX_PARTICLES 15000

class Particles
{
public:
    Particles()
    {

        cudaMallocManaged(&forces, sizeof(float3) * MX_PARTICLES);
        cudaMallocManaged(&particles, sizeof(Particle) * MX_PARTICLES);
        gen.seed(1);
        particle_size = 1.0f;
    };
    // void addParticle(float mass);
    void initParticles();
    void prepRender();
    void render(float dt);
    // void sumForces(float dt);
    __host__ void initVel();
    void initPos();
    void initSystem();
    // float3 gravitationalForce(Particle p1, Particle p2);
    float3 randCircle();

private:
    GLuint VAO;
    GLuint VBO;
    std::mt19937 gen;
    Particle *particles;
    // std::array<Particle,MX_PARTICLES> particles;
    float3 center_mass;
    float3 avg_vel;
    float3 *forces;
    float3 *pos;
    float particle_size;
};