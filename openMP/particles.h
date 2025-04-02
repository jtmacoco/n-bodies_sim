#pragma once
#include <vector>
#include <random>
#include <tuple>
#include <cmath>
#include "particle.h"
#include "omp.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <iostream>
//#define G 6.674e-11
#define G 1.0f//set to 1 because actual G constant is super small
class Particles
{
    public:
        Particles(){
            gen.seed(1);
            particle_size = 3.0f;
        };
        void addParticle(float mass);
        void prepRender();
        void render(float dt);
        void sumForces(float dt);
        void initVel();
        void initPos();
        void initSystem();
        //std::tuple<glm::vec3, glm::vec3> randCircle();
        glm::vec3 randCircle();
        glm::vec3 gravitationalForce(Particle p1, Particle p2);

    private:
        GLuint VAO; 
        GLuint VBO; 
        std::mt19937 gen;
        std::vector<std::shared_ptr<Particle> > particles;
        glm::vec3 center_mass;
        glm::vec3 avg_vel;
        float particle_size;
};