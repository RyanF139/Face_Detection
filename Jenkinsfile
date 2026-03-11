pipeline {
agent { label 'built-in' }

environment {
    REPO_URL   = 'https://github.com/RyanF139/Face_Detection.git'
    ENV_SOURCE = '/opt/config/face-detection/.env'
}

stages {

    stage('Checkout') {
        steps {
            git branch: 'main',
                url: "${REPO_URL}",
                credentialsId: '001'
        }
    }

    stage('Prepare Environment File') {
        steps {
            sh 'cp $ENV_SOURCE .env'
        }
    }

    stage('Ensure Network') {
        steps {
            sh '''
            docker network inspect shared_network >/dev/null 2>&1 || \
            docker network create shared_network
            '''
        }
    }

    stage('Stop Old Containers') {
        steps {
            sh '''
            docker stop face-detection || true
            docker rm face-detection || true
            docker compose down --remove-orphans || true
            '''
        }
    }

    stage('Build Containers') {
        steps {
            sh '''
            docker compose build --no-cache
            '''
        }
    }

    stage('Run Containers') {
        steps {
            sh '''
            docker compose up -d
            '''
        }
    }

    stage('Show Status') {
        steps {
            sh '''
            docker ps | grep face || true
            '''
        }
    }
}

post {
    success {
        echo 'Deploy sukses 🚀'
    }
    failure {
        echo 'Deploy gagal ❌'
    }
    always {
        echo 'Pipeline completed.'
    }
}

}
