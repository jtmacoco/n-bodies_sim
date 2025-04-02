#pragma once
#include <iostream>
#include <glm/glm.hpp>
#include <memory>
class Particle{
    public:
        Particle(glm::vec3 pos,glm::vec3 velocity, float mass):pos(pos),velocity(velocity),mass(mass){
            force = 0.0f;
            acceleration= glm::vec3(0);
        };
        void applyForce(glm::vec3 incoming_force, float dt);
        void setVelocity(glm::vec3 v);
        void setPosition(glm::vec3 p);
        float getMass();
        float getForce();
        glm::vec3 getAcceleration();
        glm::vec3 getVelocity();
        glm::vec3 getPosition();

    private:
        
        glm::vec3 acceleration; 
        glm::vec3 pos;
        glm::vec3 velocity;
        float mass; 
        float force;
};