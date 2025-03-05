#pragma once
#include <vector>
#include "particle.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <iostream>
#define G 6.674e-11
class Particles
{
    public:
        Particles(){};
        void addParticle(float x, float y);
        void prepRender();
        void render();
        glm::vec3 gravitationalForce(Particle p1, Particle p2);
        glm::vec3 sumForces();
    private:
        GLuint VAO; 
        GLuint VBO; 
        std::vector<std::shared_ptr<Particle> > particles;
};