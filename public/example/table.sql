Table image_events {
  session_id serial [primary key]
  user_id integer [not null]
  modification integer [not null]
  name_module varchar(100) [not null]
  ia_name varchar(100) [not null]
  quality varchar(6)
  time_taken integer
  tokens_used integer

}

Table user_created_images {
  id serial [primary key]
  user_id integer [not null]
  image_url text [not null]
  image_pathname varchar(500) [not null]
  prompt text
  image_type varchar(50) [default: 'generated']
  width integer
  height integer
  file_size bigint
  created_at timestamp [default: `CURRENT_TIMESTAMP`]
  image_event_id integer [not null] // FK a image_events.session_id

  // Constraints
  unique_user_created_image unique(user_id, image_url)
}

// Relaciones
Ref: user_created_images.image_event_id > image_events.session_id


funcion 
start 

const time = Date.now()

api

const time_end = Date.now()

return time_end - time


end