#include "particle.h"
glm::vec3 Particle::getPosition()
{
    return pos;
}
float Particle::getMass(){
    return mass;
}
void Particle::setVelocity(glm::vec3 v){
    velocity -= v;
}
glm::vec3 Particle::getAcceleration(){
    return acceleration;
}
glm::vec3 Particle::getVelocity(){
    return velocity;
}
void Particle::setPosition(glm::vec3 p){
    pos -= p;
}
float Particle::getForce(){
    return force;
}
void Particle::applyForce(glm::vec3 incoming_force, float dt){
    pos += velocity*dt;
    velocity += acceleration*dt;
    acceleration = incoming_force/mass;
}