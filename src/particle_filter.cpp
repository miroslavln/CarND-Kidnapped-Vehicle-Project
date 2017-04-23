/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <assert.h>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 100;

    default_random_engine gen;

    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    for (int i = 0; i < num_particles; i++) {
        Particle particle;
        particle.id = i;
        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_theta(gen);
        particle.weight = 1;
        particles.push_back(particle);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

    if (fabs(yaw_rate) < 1e-6)
        return;

    default_random_engine gen;
    normal_distribution<double> dist_x(0.0, std_pos[0]);
    normal_distribution<double> dist_y(0.0, std_pos[1]);
    normal_distribution<double> dist_theta(0.0, std_pos[2]);

    for (int i = 0; i < num_particles; i++) {
        Particle particle = particles[i];
        particle.x += (velocity / yaw_rate) * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
        particle.y += (velocity / yaw_rate) * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
        particle.theta += yaw_rate * delta_t;
        particle.x += dist_x(gen);
        particle.y += dist_y(gen);
        particle.theta += dist_theta(gen);
        particles[i] = particle;
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {

    for (int i = 0; i < observations.size(); i++) {
        double min_distance = 1 << 30;
        LandmarkObs obs = observations[i];
        for (int j = 0; j < predicted.size(); j++) {
            double distance = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
            if (distance < min_distance) {
                obs.id = j;
                min_distance = distance;
            }
        }
        observations[i] = obs;
    }
}

double calculateParticleWeight(const double std_landmark[], const vector<LandmarkObs> &transformed_observations,
                               const vector<LandmarkObs> &predicted_observations) {
    double weight = 0;
    for (int i = 0; i < transformed_observations.size(); i++) {
        LandmarkObs obs = transformed_observations[i];
        LandmarkObs pred = predicted_observations[obs.id];
        double diff_x = obs.x - pred.x;
        double diff_y = obs.y - pred.y;
        double sigma_x = std_landmark[0];
        double sigma_y = std_landmark[1];

        weight += -((diff_x * diff_x) / (2 * sigma_x * sigma_x) +
                    (diff_y * diff_y) / (2 * sigma_y * sigma_y) +
                    log(2 * M_PI * sigma_x * sigma_y));
    }
    return exp(weight);
}

vector<LandmarkObs> getMapObservations(const Map &map_landmarks) {
    vector<LandmarkObs> predicted_observations;
    for (int i = 0; i < map_landmarks.landmark_list.size(); i++) {
        LandmarkObs pred_obs;
        Map::single_landmark_s landmark = map_landmarks.landmark_list[i];
        pred_obs.id = -1;
        pred_obs.x = landmark.x_f;
        pred_obs.y = landmark.y_f;

        predicted_observations.push_back(pred_obs);
    }
    return predicted_observations;
}

vector<LandmarkObs> convertToMapCoordinates(const vector<LandmarkObs> &observations, const Particle &particle) {
    vector<LandmarkObs> transformed_observations;
    for (int i = 0; i < observations.size(); i++) {
        LandmarkObs obs = observations[i];
        double o_x = particle.x + obs.x * cos(particle.theta) - obs.y * sin(particle.theta);
        double o_y = particle.y + obs.x * sin(particle.theta) + obs.y * cos(particle.theta);
        obs.x = o_x;
        obs.y = o_y;
        transformed_observations.push_back(obs);
    }
    return transformed_observations;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {

    vector<LandmarkObs> predicted_observations = getMapObservations(map_landmarks);

    for (int p = 0; p < num_particles; p++) {
        Particle particle = particles[p];

        vector<LandmarkObs> transformed_observations = convertToMapCoordinates(observations, particle);
        dataAssociation(predicted_observations, transformed_observations);
        particle.weight = calculateParticleWeight(std_landmark, transformed_observations, predicted_observations);

        particles[p] = particle;
    }
}

void ParticleFilter::resample() {

    double max_weight = 0.0;
    for (int i = 0; i < num_particles; i++) {
        if (max_weight < particles[i].weight)
            max_weight = particles[i].weight;
    }

    double beta = 0.0;
    default_random_engine gen;
    uniform_int_distribution<int> sample_index(0, num_particles - 1);
    uniform_real_distribution<double> sample_beta(0.0, 2 * max_weight);
    int index = sample_index(gen);

    vector<Particle> new_particles;

    for (int i = 0; i < num_particles; i++) {
        beta += sample_beta(gen);
        while (particles[index].weight < beta) {
            beta -= particles[index].weight;
            index = (index + 1) % num_particles;
        }
        Particle particle = particles[index];
        new_particles.push_back(particle);
    }
    particles = new_particles;
}

void ParticleFilter::write(std::string filename) {
    // You don't need to modify this file.
    std::ofstream dataFile;
    dataFile.open(filename, std::ios::app);
    for (int i = 0; i < num_particles; ++i) {
        dataFile << particles[i].x << " " <<  particles[i].y << " " <<  particles[i].theta << "\n";
    }
    dataFile.close();
}
