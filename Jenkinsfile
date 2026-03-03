pipeline {
    agent any

    environment {
        APP_NAME = "face-capture"
        IMAGE_NAME = "face-capture:latest"
    }

    stages {

        stage('System Update') {
            steps {
                sh '''
                sudo apt update -y
                '''
            }
        }

        stage('Clone Repository') {
            steps {
                git 'https://github.com/RyanF139/face-detection.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh '''
                docker build -t $IMAGE_NAME .
                '''
            }
        }

        stage('Stop Old Container') {
            steps {
                sh '''
                docker stop $APP_NAME || true
                docker rm $APP_NAME || true
                '''
            }
        }

        stage('Run Container') {
            steps {
                sh '''
                docker run -d \
                  --name $APP_NAME \
                  --network host \
                  --restart unless-stopped \
                  --env-file .env \
                  -v $(pwd)/image_face:/app/image_face \
                  $IMAGE_NAME
                '''
            }
        }
    }
}