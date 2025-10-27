create database video_vector;

USE video_vector;

CREATE TABLE signs (
       id INT AUTO_INCREMENT PRIMARY KEY,
       word VARCHAR(100),
       vector JSON
       );
       
commit;

select * from signs;