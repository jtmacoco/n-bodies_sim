#pragma once
#include <iostream>
#include <glm/glm.hpp>
#include <memory>
class Particle{
    public:
        Particle(float x, float y):x(x),y(y){
            mass = 10;
            acceleration = 0;
            velocity= glm::vec3(0,0,0);
            pos = glm::vec3(x,y,0);

        };
        void update();
        float getMass();
        float getAcceleration();
        glm::vec3 getPosition();
    private:
        float x;
        float y;
        float mass; 
        float acceleration; 
        glm::vec3 pos;
        glm::vec3 velocity;
};