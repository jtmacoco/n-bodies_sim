#pragma once
#include <vector>
#include "particle.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <iostream>
class Particles
{
    public:
        Particles(){};
        void addParticle(float x, float y);
        void prepRender();
        void render();

    private:
        GLuint VAO; 
        GLuint VBO; 
        std::vector<std::shared_ptr<Partcile> > particles;
};